#!/usr/bin/env python3
"""Pull the OpenHome benchmark (https://huggingface.co/datasets/wyzelabs/OpenHome).

The HF dataset is one CSV of 2,254 questions; the 1,118 clips are not on the Hub,
only linked from it (video_url, mostly the Wyze forum CDN). This fetches the CSV
and then every clip, so evals run from local files:

    data/openhome.csv            the dataset as published (pinned with --revision)
    data/questions.jsonl         one question per line, plus video_id and video_path
    data/videos/<video_id>.mp4   the clips (video_id: sha1 of the url, 16 hex)
    data/missing.json            clips that could not be downloaded (rerun to retry)
    data/videos_640/<video_id>.mp4   with --make-640: the 640-px copies every published
                                 run was fed (long side 640, h264 crf 18, no audio,
                                 frame rate unchanged), the configs' default input

Reruns skip clips already on disk.

    python3 download.py                 # everything
    python3 download.py --make-640      # everything, plus the 640-px copies
    python3 download.py --limit 20      # the first 20 clips (smoke test)
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "wyzelabs/OpenHome"
UA = "Mozilla/5.0 (X11; Linux x86_64) openhome-eval"


def video_id(url: str) -> str:
    # Clips are keyed by url: several question_id prefixes share one clip.
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def fetch(url: str, dest: str, tries: int, timeout: float) -> None:
    tmp = dest + ".part"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if os.path.getsize(tmp) == 0:
                raise RuntimeError("empty response")
            os.replace(tmp, dest)
            return
        except Exception:
            if attempt == tries - 1:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
            time.sleep(2 ** attempt)


def make_640(ffmpeg, src, out):
    """Long side 640, h264 crf 18, no audio, frame rate unchanged."""
    tmp = out + ".partial.mp4"
    vf = "scale='if(gte(iw,ih),640,-2)':'if(gte(iw,ih),-2,640)':flags=bicubic,format=yuv420p"
    p = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-map", "0:v:0",
                        "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-an",
                        "-movflags", "+faststart", tmp], capture_output=True, text=True)
    if p.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(p.stderr.strip()[-300:])
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--revision", default="main", help="dataset commit / branch to pin")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--limit", type=int, help="only the first N clips")
    ap.add_argument("--make-640", action="store_true", help="also write the 640-px copies to data/videos_640/")
    args = ap.parse_args()

    vids_dir = os.path.join(args.data_dir, "videos")
    os.makedirs(vids_dir, exist_ok=True)
    csv_path = hf_hub_download(REPO_ID, "openhome.csv", repo_type="dataset", revision=args.revision,
                               local_dir=args.data_dir)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    urls = {}
    for r in rows:
        r["video_id"] = video_id(r["video_url"])
        r["video_path"] = os.path.join("videos", r["video_id"] + ".mp4")   # relative to data dir
        urls.setdefault(r["video_id"], r["video_url"])
    with open(os.path.join(args.data_dir, "questions.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} questions, {len(urls)} clips -> {args.data_dir}")

    todo = [(vid, url) for vid, url in urls.items()][: args.limit]
    todo = [(vid, url) for vid, url in todo if not os.path.exists(os.path.join(vids_dir, vid + ".mp4"))]
    missing = {}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, url, os.path.join(vids_dir, vid + ".mp4"), args.tries, args.timeout): (vid, url)
                for vid, url in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="clips"):
            vid, url = futs[fut]
            try:
                fut.result()
            except Exception as e:
                missing[vid] = {"url": url, "error": f"{type(e).__name__}: {e}"}

    with open(os.path.join(args.data_dir, "missing.json"), "w") as f:
        json.dump(missing, f, indent=1)
    have = sum(os.path.exists(os.path.join(vids_dir, v + ".mp4")) for v in urls)
    print(f"{have}/{len(urls)} clips on disk; {len(missing)} failed this run"
          + (" (see missing.json, rerun to retry)" if missing else ""))

    if args.make_640:
        sys.path.insert(0, HERE)
        from eval import find_ffmpeg
        ffmpeg, out_dir = find_ffmpeg(), os.path.join(args.data_dir, "videos_640")
        os.makedirs(out_dir, exist_ok=True)
        todo = [v for v in list(urls)[: args.limit] if os.path.exists(os.path.join(vids_dir, v + ".mp4"))
                and not os.path.exists(os.path.join(out_dir, v + ".mp4"))]
        failed = 0
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(make_640, ffmpeg, os.path.join(vids_dir, v + ".mp4"),
                              os.path.join(out_dir, v + ".mp4")): v for v in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="640-px copies"):
                try:
                    fut.result()
                except Exception as e:
                    failed += 1
                    tqdm.write(f"{futs[fut]}: {e}")
        print(f"{len(todo) - failed} 640-px copies made ({ffmpeg}), {failed} failed")


if __name__ == "__main__":
    main()
