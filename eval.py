#!/usr/bin/env python3
"""Evaluate a model on OpenHome from a config in configs/.

Backends (the config's `backend`):
    openai     OpenAI chat completions (GPT)
    xai        Grok through xAI's OpenAI-compatible endpoint
    gemini     Gemini generateContent: Vertex AI (vertex_project, gcloud token) or AI Studio (key)
    meta       Muse through Meta's Responses API
    anthropic  Claude through the Messages API
    vllm / sglang / openai_compatible   a local OpenAI-compatible server (serve.py)

Input (`input`):
    frames   JPEG frames at 1 fps (at least 4, at most 64, spread evenly over longer
             clips), each scaled down to <= frames.max_pixels, every frame preceded by
             its [mm:ss.sss] timestamp, then the question. Extracted once per clip and
             setting into data/frames/ and shared by every model.
    video    the clip itself: a file path / URL for vllm and sglang (the server samples
             it), inline bytes for gemini.
Clips come from data/<video_dir>/ (default videos_640, the 640-px copies every
published run used; make them with download.py --make-640).

Answer (`answer`):
    structured  a JSON field constrained to A-D (openai, xai, gemini)
    logprobs    one token, the most likely option letter among the top 20 (local servers)
    text        the first standalone option letter after any reasoning
A question with no option letter is recorded as an error and retried on rerun.

Results go to results/<name>/: predictions.jsonl (resumable), summary.json
(accuracy overall / per category / per task, unanswered counted wrong) and the
config the run used.

    python3 eval.py configs/gpt_4o.yaml --limit 20            # smoke test
    python3 eval.py configs/gpt_4o.yaml --category Insight    # the 544 Insight questions
    python3 eval.py configs/gpt_4o.yaml --dry-run --limit 3   # inputs + prompts, no calls
    python3 eval.py configs/gpt_4o.yaml --report              # summarise what is done
"""
import argparse
import base64
import copy
import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
LETTERS = ["A", "B", "C", "D"]

DEFAULTS = {
    "name": None,               # results/<name>; default: the config file's stem
    "backend": None,
    "model": None,              # local servers: None = the server's only model
    "base_url": None,           # default per backend; local servers: from server.port
    "api_key_env": None,        # default per backend (KEY_ENVS)
    "api_key_file": None,       # read the key from this file instead
    "vertex_project": None,     # gemini: use Vertex AI (location global) with `gcloud auth print-access-token`
    "input": "frames",          # frames | video
    "video_dir": "videos_640",  # under data_dir: videos_640 (as published) or videos (originals)
    "video_url": "file://{path}",   # input=video on a local server; {path} = the clip's absolute path
    "frames": {"fps": 1.0, "min_frames": 4, "max_frames": 64, "max_pixels": 200_000, "jpeg_quality": 3},
    "timestamps": True,         # a [mm:ss.sss] text part before each frame
    "instruction": None,        # a text part before the frames
    "image_detail": None,       # openai / xai / meta: low | high | auto
    "prompt": ("{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n"
               "Answer with the option letter only."),   # "as written": the text every published run used
    "answer": "text",           # structured | logprobs | text
    "generation": {},           # passed to the API call as is (gemini: its generationConfig)
    "extra_body": None,         # openai-compatible only, e.g. chat_template_kwargs
    "concurrency": 16,
    "timeout": 600,
    "max_retries": 6,           # 429 / 5xx / timeouts, with backoff
    "data_dir": "data",
    "output_dir": "results",
    "server": None,             # serve.py: how to launch the local server
    "notes": None,
}

BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "xai": "https://api.x.ai/v1",
    "meta": "https://api.meta.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}
KEY_ENVS = {"openai": "OPENAI_API_KEY", "xai": "XAI_API_KEY", "meta": "META_API_KEY", "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", "vllm": "VLLM_API_KEY", "sglang": "SGLANG_API_KEY",
            "openai_compatible": "OPENAI_API_KEY"}
LOCAL = {"vllm", "sglang", "openai_compatible"}


# --------------------------------------------------------------------- config

def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, overrides=None):
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    unknown = set(user) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)}")
    cfg = merge(DEFAULTS, user)
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    b = cfg["backend"]
    if b not in KEY_ENVS:
        raise SystemExit(f"{path}: backend must be one of {sorted(KEY_ENVS)}")
    if cfg["input"] == "video" and b not in LOCAL | {"gemini"}:
        raise SystemExit(f"{path}: input video is for local servers and gemini only")
    if cfg["answer"] == "structured" and b not in {"openai", "xai", "gemini"}:
        raise SystemExit(f"{path}: answer structured is for openai / xai / gemini")
    if cfg["answer"] == "logprobs" and b not in LOCAL | {"openai"}:
        raise SystemExit(f"{path}: answer logprobs needs an OpenAI-compatible chat API")
    cfg["name"] = cfg["name"] or os.path.splitext(os.path.basename(path))[0]
    if not cfg["base_url"]:
        port = (cfg["server"] or {}).get("port")
        cfg["base_url"] = f"http://localhost:{port}/v1" if b in LOCAL and port else BASE_URLS.get(b)
    for k in ("data_dir", "output_dir"):
        cfg[k] = cfg[k] if os.path.isabs(cfg[k]) else os.path.join(HERE, cfg[k])
    return cfg


def api_key(cfg):
    if cfg["api_key_file"]:
        with open(os.path.expanduser(cfg["api_key_file"])) as f:
            return f.read().strip()
    env = cfg["api_key_env"] or KEY_ENVS[cfg["backend"]]
    key = os.environ.get(env)
    if not key and cfg["backend"] not in LOCAL and not cfg["vertex_project"]:
        raise SystemExit(f"no API key: set {env} or api_key_file")
    return key or "EMPTY"


# --------------------------------------------------------------------- frames

def find_ffmpeg():
    exe = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found: install it, set FFMPEG, or pip install imageio-ffmpeg")


def probe_duration(ffmpeg, path):
    """Seconds, from the container header, else by decoding the whole clip."""
    p = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", p.stderr)
    if m:
        return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    p = subprocess.run([ffmpeg, "-hide_banner", "-i", path, "-map", "0:v:0", "-f", "null", "-"],
                       capture_output=True, text=True)
    times = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", p.stderr)
    if not times:
        raise RuntimeError(f"cannot read the duration of {path}")
    h, m_, s = times[-1]
    return int(h) * 3600 + int(m_) * 60 + float(s)


class Frames:
    """[(seconds, jpeg bytes)] per clip, cached on disk under data/frames/<setting>/<video_id>/."""

    def __init__(self, cfg):
        f = cfg["frames"]
        self.fps, self.lo, self.hi, self.px, self.q = f["fps"], f["min_frames"], f["max_frames"], f["max_pixels"], f["jpeg_quality"]
        self.root = os.path.join(cfg["data_dir"], "frames",
                                 f"{cfg['video_dir']}_fps{self.fps}_n{self.lo}-{self.hi}_px{self.px}_q{self.q}")
        self.ffmpeg = find_ffmpeg()
        self.locks = defaultdict(threading.Lock)

    def get(self, video_id, path):
        out = os.path.join(self.root, video_id)
        with self.locks[video_id]:
            meta = os.path.join(out, "times.json")
            if not os.path.exists(meta):
                self.extract(path, out)
            with open(meta) as f:
                times = json.load(f)
            files = sorted(glob.glob(os.path.join(out, "f_*.jpg")))
            return [(t, open(fn, "rb").read()) for t, fn in zip(times, files)]

    def extract(self, path, out):
        dur = probe_duration(self.ffmpeg, path)
        n = min(max(int(dur * self.fps), self.lo), self.hi)
        rate = n / dur if dur > 0 else self.fps
        k = f"min(1,sqrt({self.px}/(iw*ih)))"
        vf = f"fps={rate:.6f},scale='trunc(iw*{k}/2)*2':'trunc(ih*{k}/2)*2'"
        tmp = out + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        p = subprocess.run([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path, "-vf", vf,
                            "-frames:v", str(n), "-q:v", str(self.q), os.path.join(tmp, "f_%04d.jpg")],
                           capture_output=True, text=True)
        files = sorted(glob.glob(os.path.join(tmp, "f_*.jpg")))
        if not files:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError(f"no frames decoded from {path}: {p.stderr.strip()[-300:]}")
        with open(os.path.join(tmp, "times.json"), "w") as f:
            json.dump([i / rate for i in range(len(files))], f)
        shutil.rmtree(out, ignore_errors=True)
        os.replace(tmp, out)


def stamp(t):
    return f"[{int(t // 60):02d}:{t - 60 * int(t // 60):06.3f}]"


def b64(data):
    return base64.b64encode(data).decode()


# --------------------------------------------------------------------- answer

THINK = re.compile(r"<think>.*?</think>", re.S)
LETTER = re.compile(r"\b([ABCD])\b")


def parse_letter(text):
    """The first standalone option letter after any reasoning, as the published runs parsed it."""
    text = THINK.sub("", text or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    m = LETTER.search(text)
    return m.group(1) if m else None


class NoAnswer(RuntimeError):
    pass


def letter_or_raise(text, finish=None):
    pred = parse_letter(text)
    if not pred:
        raise NoAnswer(f"no option letter in the answer {(text or '')[-200:]!r} (finish={finish})")
    return pred


SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"answer": {"type": "string", "enum": LETTERS}}, "required": ["answer"]}


# ------------------------------------------------------------------- backends

class OpenAIChat:
    """openai / xai / vllm / sglang / openai_compatible chat completions."""

    def __init__(self, cfg):
        from openai import OpenAI
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg["base_url"], api_key=api_key(cfg),
                             timeout=cfg["timeout"], max_retries=cfg["max_retries"])
        self.model = cfg["model"]
        if not self.model:
            if cfg["backend"] not in LOCAL:
                raise SystemExit("model is required for " + cfg["backend"])
            self.model = self.client.models.list().data[0].id

    def content(self, media, text):
        parts = [{"type": "text", "text": self.cfg["instruction"]}] if self.cfg["instruction"] else []
        if isinstance(media, str):                       # input: video
            parts.append({"type": "video_url", "video_url": {"url": media}})
        else:
            for t, jpg in media:
                if self.cfg["timestamps"]:
                    parts.append({"type": "text", "text": stamp(t)})
                img = {"url": "data:image/jpeg;base64," + b64(jpg)}
                if self.cfg["image_detail"]:
                    img["detail"] = self.cfg["image_detail"]
                parts.append({"type": "image_url", "image_url": img})
        parts.append({"type": "text", "text": text})
        return parts

    def ask(self, media, text):
        gen = dict(self.cfg["generation"])
        mode = self.cfg["answer"]
        if mode == "structured":
            gen["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "mcq_answer", "strict": True, "schema": SCHEMA}}
        elif mode == "logprobs":
            gen.update(max_tokens=1, logprobs=True, top_logprobs=20)
        r = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": self.content(media, text)}],
            extra_body=self.cfg["extra_body"], **gen)
        ch = r.choices[0]
        msg = ch.message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        out = {"raw": msg.content or "", "finish_reason": ch.finish_reason,
               "reasoning_chars": len(reasoning or ""), "usage": r.usage.model_dump() if r.usage else None}
        if mode == "structured":
            if getattr(msg, "refusal", None):
                raise RuntimeError(f"refused: {msg.refusal}")
            out["pred"] = json.loads(msg.content)["answer"]
        elif mode == "logprobs":
            lps = {}
            top = ch.logprobs.content[0].top_logprobs if ch.logprobs and ch.logprobs.content else []
            for c in top:
                k = (c.token or "").strip()
                if k in LETTERS and c.logprob > lps.get(k, float("-inf")):
                    lps[k] = round(c.logprob, 4)
            if not lps:
                raise NoAnswer(f"no option letter in the top-20 (said {msg.content!r})")
            out.update(pred=max(lps, key=lps.get), logprobs=lps)
        else:
            out["pred"] = letter_or_raise(msg.content, ch.finish_reason)
        return out


class MetaResponses:
    """Meta's Responses API (Muse Wonder)."""

    def __init__(self, cfg):
        from openai import OpenAI
        if not cfg["model"]:
            raise SystemExit("model is required for meta")
        self.cfg, self.model = cfg, cfg["model"]
        self.client = OpenAI(base_url=cfg["base_url"], api_key=api_key(cfg),
                             timeout=cfg["timeout"], max_retries=cfg["max_retries"])

    def ask(self, frames, text):
        parts = [{"type": "input_text", "text": self.cfg["instruction"]}] if self.cfg["instruction"] else []
        for t, jpg in frames:
            if self.cfg["timestamps"]:
                parts.append({"type": "input_text", "text": stamp(t)})
            img = {"type": "input_image", "image_url": "data:image/jpeg;base64," + b64(jpg)}
            if self.cfg["image_detail"]:
                img["detail"] = self.cfg["image_detail"]
            parts.append(img)
        parts.append({"type": "input_text", "text": text})
        r = self.client.responses.create(model=self.model, input=[{"role": "user", "content": parts}],
                                         **self.cfg["generation"])
        out = r.output_text.strip()
        usage = r.usage.model_dump() if r.usage else None
        return {"raw": out, "finish_reason": r.status, "usage": usage,
                "pred": letter_or_raise(out, r.status)}


class GcloudToken:
    """A gcloud user access token, refreshed well before it expires (tokens last an hour)."""

    def __init__(self, ttl=2400):
        self.ttl, self.lock, self.value, self.t = ttl, threading.Lock(), None, 0

    def get(self):
        with self.lock:
            if not self.value or time.time() - self.t > self.ttl:
                self.value = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True,
                                            text=True, check=True).stdout.strip()
                self.t = time.time()
            return self.value


class GeminiNative:
    """generateContent over REST: Vertex AI (location global) or AI Studio."""

    def __init__(self, cfg):
        if not cfg["model"]:
            raise SystemExit("model is required for gemini")
        self.cfg, self.model = cfg, cfg["model"]
        if cfg["vertex_project"]:
            self.url = (f"https://aiplatform.googleapis.com/v1/projects/{cfg['vertex_project']}"
                        f"/locations/global/publishers/google/models/{self.model}:generateContent")
            token = GcloudToken()
            self.headers = lambda: {"Authorization": f"Bearer {token.get()}"}
        else:
            self.url = f"{cfg['base_url']}/models/{self.model}:generateContent"
            key = api_key(cfg)
            self.headers = lambda: {"x-goog-api-key": key}

    def post(self, body):
        data = json.dumps(body).encode()
        for attempt in range(self.cfg["max_retries"] + 1):
            req = urllib.request.Request(self.url, data=data, method="POST",
                                         headers={"Content-Type": "application/json", **self.headers()})
            try:
                with urllib.request.urlopen(req, timeout=self.cfg["timeout"]) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}: {e.read()[:300]!r}"
                if e.code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(err) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                err = f"{type(e).__name__}: {e}"
            if attempt == self.cfg["max_retries"]:
                raise RuntimeError(err)
            time.sleep(min(60, 2 ** attempt) + attempt)

    def ask(self, media, text):
        parts = [{"text": self.cfg["instruction"]}] if self.cfg["instruction"] else []
        if isinstance(media, bytes):                     # input: video, inline
            parts.append({"inline_data": {"mime_type": "video/mp4", "data": b64(media)}})
        else:
            for t, jpg in media:
                if self.cfg["timestamps"]:
                    parts.append({"text": stamp(t)})
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64(jpg)}})
        parts.append({"text": text})
        gen = dict(self.cfg["generation"])
        if self.cfg["answer"] == "structured":
            gen.update(responseMimeType="application/json", responseSchema={
                "type": "OBJECT", "properties": {"answer": {"type": "STRING", "enum": LETTERS}},
                "required": ["answer"]})
        r = self.post({"contents": [{"role": "user", "parts": parts}], "generationConfig": gen})
        cands = r.get("candidates") or []
        if not cands:
            raise RuntimeError(f"no candidates: {json.dumps(r.get('promptFeedback'))[:300]}")
        txt = "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts") or []
                      if not p.get("thought"))
        finish = cands[0].get("finishReason")
        if not txt:
            raise RuntimeError(f"empty answer, finishReason={finish}")
        pred = json.loads(txt)["answer"] if self.cfg["answer"] == "structured" else letter_or_raise(txt, finish)
        return {"raw": txt, "finish_reason": finish, "usage": r.get("usageMetadata"), "pred": pred}


class AnthropicChat:
    def __init__(self, cfg):
        import anthropic
        if not cfg["model"]:
            raise SystemExit("model is required for anthropic")
        self.cfg, self.model = cfg, cfg["model"]
        kw = {"base_url": cfg["base_url"]} if cfg["base_url"] else {}
        self.client = anthropic.Anthropic(api_key=api_key(cfg), timeout=cfg["timeout"],
                                          max_retries=cfg["max_retries"], **kw)

    def ask(self, frames, text):
        parts = [{"type": "text", "text": self.cfg["instruction"]}] if self.cfg["instruction"] else []
        for t, jpg in frames:
            if self.cfg["timestamps"]:
                parts.append({"type": "text", "text": stamp(t)})
            parts.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(jpg)}})
        parts.append({"type": "text", "text": text})
        r = self.client.messages.create(model=self.model, messages=[{"role": "user", "content": parts}],
                                        **self.cfg["generation"])
        out = "".join(b.text for b in r.content if b.type == "text")
        return {"raw": out, "finish_reason": r.stop_reason, "usage": r.usage.model_dump() if r.usage else None,
                "pred": letter_or_raise(out, r.stop_reason)}


def make_backend(cfg):
    return {"anthropic": AnthropicChat, "gemini": GeminiNative, "meta": MetaResponses}.get(
        cfg["backend"], OpenAIChat)(cfg)


# ----------------------------------------------------------------------- run

def load_questions(cfg):
    path = os.path.join(cfg["data_dir"], "questions.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run download.py first")
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_results(path):
    """question_id -> the last result without an error, else the last error."""
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("error") or done.get(r["question_id"], {}).get("error", True):
                    done[r["question_id"]] = r
    return done


def summarise(questions, results):
    """Accuracy over every selected question; no answer counts wrong."""
    def acc(qs):
        ok = sum(bool(results.get(q["question_id"], {}).get("correct")) for q in qs)
        return {"n": len(qs), "correct": ok, "accuracy": round(100 * ok / len(qs), 2) if qs else None}
    by_cat, by_type = defaultdict(list), defaultdict(list)
    for q in questions:
        by_cat[q["category"]].append(q)
        by_type[q["type"]].append(q)
    answered = [q for q in questions if q["question_id"] in results and not results[q["question_id"]].get("error")]
    return {"questions": len(questions), "answered": len(answered),
            "errors": sum(1 for q in questions if results.get(q["question_id"], {}).get("error")),
            "overall": acc(questions), "category": {k: acc(v) for k, v in sorted(by_cat.items())},
            "type": {k: acc(v) for k, v in sorted(by_type.items())}}


def print_summary(name, s):
    print(f"\n{name}: {s['answered']}/{s['questions']} answered ({s['errors']} errors; unanswered count wrong)")
    order = ["Perception", "Reasoning", "Insight"]
    for k in sorted(s["category"], key=lambda c: order.index(c) if c in order else 9):
        print(f"  {k:<12} {s['category'][k]['accuracy']:6.2f}  (n={s['category'][k]['n']})")
    print(f"  {'Overall':<12} {s['overall']['accuracy'] or 0:6.2f}")
    for k, v in s["type"].items():
        print(f"    {k:<12} {v['accuracy']:6.2f}  (n={v['n']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--name", help="results/<name> (default: the config's name or file stem)")
    ap.add_argument("--model"), ap.add_argument("--base-url")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--category", help="comma list: Perception, Reasoning, Insight")
    ap.add_argument("--type", help="comma list of task types, e.g. Intent,Evidence")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--dry-run", action="store_true", help="build the inputs, print prompts, no calls")
    ap.add_argument("--report", action="store_true", help="summarise existing results and exit")
    args = ap.parse_args()

    cfg = load_config(args.config, {"name": args.name, "model": args.model, "base_url": args.base_url,
                                    "concurrency": args.concurrency})
    out_dir = os.path.join(cfg["output_dir"], cfg["name"])
    pred_path = os.path.join(out_dir, "predictions.jsonl")
    questions = load_questions(cfg)
    if args.category:
        questions = [q for q in questions if q["category"] in args.category.split(",")]
    if args.type:
        questions = [q for q in questions if q["type"] in args.type.split(",")]
    questions = questions[: args.limit]

    if args.report:
        print_summary(cfg["name"], summarise(questions, load_results(pred_path)))
        return

    vdir = os.path.join(cfg["data_dir"], cfg["video_dir"])
    clip = lambda q: os.path.join(vdir, q["video_id"] + ".mp4")
    have = [q for q in questions if os.path.exists(clip(q))]
    if len(have) < len(questions):
        print(f"skipping {len(questions) - len(have)} questions with no clip in {vdir} "
              "(download.py, --make-640 for videos_640)")
    results = load_results(pred_path)
    todo = sorted((q for q in have if results.get(q["question_id"], {"error": 1}).get("error")),
                  key=lambda q: q["video_id"])
    print(f"{cfg['name']}: {len(questions) - len(todo)} done, {len(todo)} to run")

    frames = Frames(cfg) if cfg["input"] == "frames" else None

    def media(q):
        if frames:
            return frames.get(q["video_id"], clip(q))
        if cfg["backend"] == "gemini":
            with open(clip(q), "rb") as f:
                return f.read()
        return cfg["video_url"].format(path=os.path.abspath(clip(q)), video_id=q["video_id"], url=q["video_url"])

    def text(q):
        return cfg["prompt"].format(question=q["question"], **{L: q["option_" + L] for L in LETTERS})

    if args.dry_run:
        for q in todo[:5]:
            m = media(q)
            desc = (m if isinstance(m, str) else f"video, {len(m) / 1e6:.2f} MB inline" if isinstance(m, bytes)
                    else f"{len(m)} frames, {sum(len(j) for _, j in m) / 1e6:.2f} MB jpeg")
            print(f"\n{q['question_id']}: {desc}\n{text(q)}")
        return

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    backend = make_backend(cfg)
    print(f"{cfg['backend']} {backend.model} at {cfg['base_url'] if not cfg['vertex_project'] else 'vertex ai'}")
    write_lock = threading.Lock()

    def run(q):
        rec = {k: q[k] for k in ("question_id", "video_id", "category", "type", "ground_truth")}
        t0 = time.time()
        try:
            m = media(q)
            if isinstance(m, list):
                rec["num_frames"] = len(m)
            rec.update(backend.ask(m, text(q)))
            rec["correct"] = rec["pred"] == q["ground_truth"]
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:1000]
        rec["seconds"] = round(time.time() - t0, 2)
        with write_lock, open(pred_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    errors = 0
    with ThreadPoolExecutor(cfg["concurrency"]) as ex:
        futs = [ex.submit(run, q) for q in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=cfg["name"]):
            rec = fut.result()
            if rec.get("error"):
                errors += 1
                if errors <= 5:
                    tqdm.write(f"{rec['question_id']}: {rec['error'][:300]}")

    # Rewrite with one line per question (the latest good one), in dataset order.
    results = load_results(pred_path)
    with open(pred_path, "w") as f:
        for q in load_questions(cfg):
            if q["question_id"] in results:
                f.write(json.dumps(results[q["question_id"]]) + "\n")

    s = summarise(questions, results)
    s.update({"name": cfg["name"], "backend": cfg["backend"], "model": backend.model,
              "filter": {"category": args.category, "type": args.type, "limit": args.limit}})
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    print_summary(cfg["name"], s)
    if s["errors"]:
        print(f"{s['errors']} questions failed (`error` in predictions.jsonl); rerun to retry them")


if __name__ == "__main__":
    main()
