#!/usr/bin/env python3
"""Launch the vLLM / SGLang server a config's `server:` block describes, in docker, the way the
published runs served each open model (pinned image, tensor parallel, multimodal settings).

    server:
      engine: sglang                  # sglang | vllm
      image: lmsysorg/sglang:latest   # the docker image
      tp: 1                           # tensor parallel size
      port: 8301                      # host port (eval.py's base_url defaults to it)
      shm: 32g
      args: [--context-length, "32768", ...]   # engine flags; "@file.json" is replaced by
                                               # that file's JSON (paths relative to the config)

The model is served under the config's `model` name, with data/ mounted read-only at the same
path inside the container (so eval.py's file paths work as they are) and the HF cache from
$HF_CACHE (default ~/.cache/huggingface). Set HF_HUB_OFFLINE=1 to forbid downloads.

    python3 serve.py configs/qwen3_5_27b.yaml start --gpus 0
    python3 serve.py configs/qwen3_5_27b.yaml wait
    python3 serve.py configs/qwen3_5_27b.yaml stop
    python3 serve.py configs/qwen3_5_27b.yaml run --gpus 0 [-- eval.py args]   # start, wait, eval, stop
    python3 serve.py configs/qwen3_5_27b.yaml command --gpus 0                 # print the docker command
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request

from eval import HERE, load_config


def container(cfg):
    return "openhome-" + "".join(c if c.isalnum() or c in "-_." else "-" for c in cfg["name"])


def engine_args(cfg, config_path):
    srv = cfg["server"]
    out = []
    for a in srv.get("args") or []:
        a = str(a)
        if a.startswith("@"):
            p = os.path.join(os.path.dirname(os.path.abspath(config_path)), a[1:])
            with open(p) as f:
                a = json.dumps(json.load(f), separators=(",", ":"))
        out.append(a)
    return out


def docker_cmd(cfg, config_path, gpus):
    srv = cfg["server"]
    port, tp = srv["port"], srv.get("tp", 1)
    hf_cache = os.path.expanduser(os.environ.get("HF_CACHE", "~/.cache/huggingface"))
    data = cfg["data_dir"]
    inner = 30000 if srv["engine"] == "sglang" else 8000
    cmd = ["docker", "run", "-d", "--name", container(cfg), "--gpus", f'"device={gpus}"', "--ipc=host",
           f"--shm-size={srv.get('shm', '32g')}", "-p", f"127.0.0.1:{port}:{inner}",
           "-v", f"{data}:{data}:ro", "-v", f"{hf_cache}:/root/.cache/huggingface"]
    for env in ("HF_TOKEN", "HF_HUB_OFFLINE"):
        if os.environ.get(env):
            cmd += ["-e", f"{env}={os.environ[env]}"]
    cmd.append(srv["image"])
    if srv["engine"] == "sglang":
        cmd += ["python3", "-m", "sglang.launch_server", "--model-path", cfg["model"],
                "--served-model-name", cfg["model"], "--host", "0.0.0.0", "--port", str(inner), "--tp-size", str(tp)]
    elif srv["engine"] == "vllm":     # the vllm-openai image's entrypoint is the server
        cmd += ["--model", cfg["model"], "--served-model-name", cfg["model"], "--host", "0.0.0.0",
                "--port", str(inner), "--tensor-parallel-size", str(tp), "--allowed-local-media-path", data]
    else:
        raise SystemExit(f"server.engine must be sglang or vllm, not {srv['engine']!r}")
    return cmd + engine_args(cfg, config_path)


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def running(name):
    out = subprocess.run(["docker", "ps", "-q", "-f", f"name=^{name}$"], capture_output=True, text=True).stdout
    return bool(out.strip())


def wait(cfg, timeout_min=90):
    name, port = container(cfg), cfg["server"]["port"]
    print(f"waiting for {name} on :{port}", end="", flush=True)
    for _ in range(timeout_min * 12):
        if healthy(port):
            print(" ready")
            return True
        if not running(name):
            print(" exited:")
            subprocess.run(["docker", "logs", "--tail", "40", name])
            return False
        print(".", end="", flush=True)
        time.sleep(5)
    print(" timed out")
    return False


def stop(cfg):
    name = container(cfg)
    log = os.path.join(cfg["output_dir"], cfg["name"], "server.log")
    if running(name):
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as f:
            subprocess.run(["docker", "logs", name], stdout=f, stderr=subprocess.STDOUT)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    print(f"stopped {name} (log: {log})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("action", choices=["start", "wait", "stop", "run", "command"])
    ap.add_argument("--gpus", help="e.g. 0 or 0,1,2,3 (needs server.tp of them)")
    argv = sys.argv[1:]
    eval_args = argv[argv.index("--") + 1:] if "--" in argv else []
    args = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
    cfg = load_config(args.config)
    if not cfg["server"]:
        raise SystemExit(f"{args.config} has no server: block")
    if args.action in ("start", "run", "command"):
        if not args.gpus:
            raise SystemExit("--gpus is required")
        cmd = docker_cmd(cfg, args.config, args.gpus)
        if args.action == "command":
            print(shlex.join(cmd))
            return
        subprocess.run(["docker", "rm", "-f", container(cfg)], capture_output=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        print(f"{container(cfg)}: {cfg['model']} gpus={args.gpus} tp={cfg['server'].get('tp', 1)} "
              f"port={cfg['server']['port']} image={cfg['server']['image']}")
    if args.action == "wait":
        sys.exit(0 if wait(cfg) else 1)
    if args.action == "stop":
        stop(cfg)
    if args.action == "run":
        ok = wait(cfg)
        try:
            if ok:
                subprocess.run([sys.executable, os.path.join(HERE, "eval.py"), args.config] + eval_args)
        finally:
            stop(cfg)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
