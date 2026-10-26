<p align="center">
  <img src="assets/openhome_logo.jpg" alt="OpenHome" width="600">
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/wyzelabs/OpenHome"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?logo=huggingface&logoColor=000" alt="Hugging Face dataset"></a>
  <a href="http://arxiv.org/abs/2610.31986"><img src="https://img.shields.io/badge/arXiv-2610.31986-B31B1B?logo=arxiv&logoColor=white" alt="arXiv"></a>
</p>

# OpenHome

Evaluation harness for [OpenHome](https://huggingface.co/datasets/wyzelabs/OpenHome): 2,254 four-option video QA questions on 1,118 smart-home camera clips, grouped into Perception, Reasoning and Insight.

There is one config per model in `configs/`. API models (OpenAI, Gemini, xAI, Meta, Anthropic) are called directly. Open models run on vLLM or SGLang, started in docker by `serve.py`.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python download.py --make-640     # questions, clips and 640-px copies -> data/
```

## Run

API models:

```bash
export OPENAI_API_KEY=...    # or XAI_API_KEY, META_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY
.venv/bin/python eval.py configs/gpt_4o.yaml                       # all questions
.venv/bin/python eval.py configs/gpt_4o.yaml --category Insight    # one category
.venv/bin/python eval.py configs/gpt_4o.yaml --report              # print scores
```

Open models:

```bash
export HF_CACHE=/path/to/hf_cache    # default ~/.cache/huggingface
.venv/bin/python serve.py configs/qwen3_5_27b.yaml run --gpus 0                        # all questions
.venv/bin/python serve.py configs/qwen3_5_27b.yaml run --gpus 0 -- --category Insight  # one category
.venv/bin/python eval.py  configs/qwen3_5_27b.yaml --report                            # print scores
```

`run` starts the server, runs the eval and stops the server. Pass `server.tp` GPUs.

## Results

Each run writes to `results/<name>/`:

- `predictions.jsonl`: one line per question. Runs resume, and failed questions are retried.
- `summary.json`: accuracy overall, per category and per task. Questions with no answer count as wrong.
- `config.yaml`: the config used.

## Config keys

| key | meaning |
|---|---|
| `backend` | `openai`, `xai`, `gemini`, `meta`, `anthropic`, `vllm`, `sglang`, `openai_compatible` |
| `model`, `base_url` | model id and endpoint |
| `api_key_env`, `api_key_file`, `vertex_project` | credentials |
| `input` | `frames` (sampled JPEGs) or `video` (the clip) |
| `frames` | `fps`, `min_frames`, `max_frames`, `max_pixels` |
| `answer` | `structured` (JSON A–D), `logprobs` (top letter) or `text` (first letter) |
| `generation`, `extra_body` | passed to the API call unchanged |
| `server` | docker `engine`, `image`, `tp`, `port`, `args` for `serve.py` |
| `concurrency`, `timeout`, `max_retries` | request handling |
