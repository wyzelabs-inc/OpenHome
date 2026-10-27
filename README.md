<p align="center">
  <img src="assets/openhome_logo.jpg" alt="OpenHome" width="600">
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/wyzelabs/OpenHome"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?logo=huggingface&logoColor=000" alt="Hugging Face dataset"></a>
  <a href="http://arxiv.org/abs/2610.31986"><img src="https://img.shields.io/badge/arXiv-2610.31986-B31B1B?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg" alt="License: CC BY-NC 4.0"></a>
</p>

# OpenHome

Evaluation harness for [OpenHome](https://huggingface.co/datasets/wyzelabs/OpenHome): 2,254 four-option video QA questions on 1,118 smart-home camera clips, grouped into Perception, Reasoning and Insight. The benchmark is described in the paper [OpenHome: From Perception to Insight in Smart-Home Camera Videos](https://arxiv.org/abs/2610.31986).

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

## Leaderboard

Accuracy (%) on all 2,254 questions. Click a model to expand its task scores. Best in **bold**, second <ins>underlined</ins>.

| # | Model | Perception | Reasoning | Insight | Overall |
|--:|---|:-:|:-:|:-:|:-:|
| 1 | <details><summary>Gemini 4 Argon</summary><i>Perception:</i> Attribute <b>89.2</b> · Recognition <b>78.5</b> · Change <b>92.4</b> · Count <b>66.1</b><br><i>Reasoning:</i> Causality <b>86.3</b> · Spatial <b>71.7</b> · Temporal <b>65.7</b><br><i>Insight:</i> Intent <b>87.2</b> · Evidence <b>90.9</b> · Response <b>89.5</b></details> | **81.6** | **74.8** | **89.2** | **81.2** |
| 2 | <details><summary>GPT-6 Astra</summary><i>Perception:</i> Attribute 83.9 · Recognition <ins>72.8</ins> · Change 87.2 · Count <ins>61.6</ins><br><i>Reasoning:</i> Causality <ins>79.0</ins> · Spatial <ins>70.5</ins> · Temporal <ins>56.8</ins><br><i>Insight:</i> Intent <ins>84.5</ins> · Evidence <ins>89.8</ins> · Response <ins>86.5</ins></details> | <ins>76.4</ins> | <ins>68.9</ins> | <ins>86.9</ins> | <ins>76.6</ins> |
| 3 | <details><summary>GPT-6.1 Sol</summary><i>Perception:</i> Attribute 82.7 · Recognition 68.7 · Change <ins>87.6</ins> · Count 55.9<br><i>Reasoning:</i> Causality 78.2 · Spatial 67.9 · Temporal 54.7<br><i>Insight:</i> Intent 78.1 · Evidence 86.6 · Response 80.1</details> | 73.8 | 67.1 | 81.6 | 73.6 |
| 4 | <details><summary>Gemini 3.8 Flash</summary><i>Perception:</i> Attribute 82.3 · Recognition 65.5 · Change 84.7 · Count 50.6<br><i>Reasoning:</i> Causality 75.8 · Spatial 61.6 · Temporal 51.3<br><i>Insight:</i> Intent 80.2 · Evidence 79.6 · Response 77.2</details> | 70.9 | 63.1 | 79.0 | 70.4 |
| 5 | <details><summary>Muse Wonder 1.0</summary><i>Perception:</i> Attribute <ins>84.3</ins> · Recognition 61.4 · Change 81.9 · Count 51.4<br><i>Reasoning:</i> Causality 73.0 · Spatial 63.3 · Temporal 49.6<br><i>Insight:</i> Intent 68.4 · Evidence 78.5 · Response 74.3</details> | 69.9 | 62.1 | 73.7 | 68.3 |
| 6 | <details><summary>Gemini 3.1 Pro</summary><i>Perception:</i> Attribute 79.9 · Recognition 58.5 · Change 75.1 · Count 46.1<br><i>Reasoning:</i> Causality 73.4 · Spatial 59.5 · Temporal 45.8<br><i>Insight:</i> Intent 64.7 · Evidence 68.8 · Response 66.7</details> | 65.0 | 59.8 | 66.7 | 63.8 |
| 7 | <details><summary>Gemini 3.5 Flash</summary><i>Perception:</i> Attribute 83.1 · Recognition 61.8 · Change 73.5 · Count 46.5<br><i>Reasoning:</i> Causality 66.5 · Spatial 53.6 · Temporal 44.5<br><i>Insight:</i> Intent 61.0 · Evidence 68.3 · Response 64.3</details> | 66.3 | 55.1 | 64.5 | 62.3 |
| 8 | <details><summary>Gemini 2.5 Pro</summary><i>Perception:</i> Attribute 80.3 · Recognition 59.4 · Change 73.5 · Count 40.4<br><i>Reasoning:</i> Causality 66.9 · Spatial 46.0 · Temporal 43.2<br><i>Insight:</i> Intent 52.9 · Evidence 59.7 · Response 61.4</details> | 63.5 | 52.3 | 57.9 | 58.6 |
| 9 | <details><summary>Muse-Glimmer-30B</summary><i>Perception:</i> Attribute 81.9 · Recognition 55.3 · Change 77.1 · Count 46.5<br><i>Reasoning:</i> Causality 63.3 · Spatial 50.2 · Temporal 43.2<br><i>Insight:</i> Intent 48.1 · Evidence 52.2 · Response 55.0</details> | 65.3 | 52.4 | 51.7 | 57.9 |
| 10 | <details><summary>GPT-6 Luna</summary><i>Perception:</i> Attribute 78.7 · Recognition 46.7 · Change 73.9 · Count 38.0<br><i>Reasoning:</i> Causality 66.5 · Spatial 48.9 · Temporal 43.6<br><i>Insight:</i> Intent 55.1 · Evidence 67.2 · Response 57.9</details> | 59.4 | 53.2 | 60.1 | 57.6 |
| 11 | <details><summary>Grok 4.7</summary><i>Perception:</i> Attribute 77.1 · Recognition 45.9 · Change 74.7 · Count 40.4<br><i>Reasoning:</i> Causality 64.1 · Spatial 50.6 · Temporal 45.3<br><i>Insight:</i> Intent 50.8 · Evidence 71.0 · Response 50.3</details> | 59.6 | 53.5 | 57.5 | 57.2 |
| 12 | <details><summary>Gemini 3.5 Flash-Lite</summary><i>Perception:</i> Attribute 77.1 · Recognition 55.3 · Change 65.9 · Count 42.9<br><i>Reasoning:</i> Causality 67.7 · Spatial 43.0 · Temporal 34.8<br><i>Insight:</i> Intent 51.3 · Evidence 54.3 · Response 56.7</details> | 60.4 | 48.8 | 54.0 | 55.2 |
| 13 | <details><summary>Gemma 4 31B-it</summary><i>Perception:</i> Attribute 78.7 · Recognition 45.9 · Change 79.9 · Count 43.3<br><i>Reasoning:</i> Causality 63.7 · Spatial 47.7 · Temporal 42.4<br><i>Insight:</i> Intent 46.0 · Evidence 44.1 · Response 46.1</details> | 62.1 | 51.5 | 45.4 | 54.7 |
| 14 | <details><summary>GPT-4o</summary><i>Perception:</i> Attribute 74.3 · Recognition 52.8 · Change 66.7 · Count 33.1<br><i>Reasoning:</i> Causality 65.7 · Spatial 37.1 · Temporal 38.1<br><i>Insight:</i> Intent 51.3 · Evidence 60.8 · Response 55.0</details> | 56.8 | 47.3 | 55.7 | 53.5 |
| 15 | <details><summary>Qwen3.5-397B-A17B</summary><i>Perception:</i> Attribute 76.3 · Recognition 44.7 · Change 71.9 · Count 32.7<br><i>Reasoning:</i> Causality 62.1 · Spatial 38.4 · Temporal 40.3<br><i>Insight:</i> Intent 45.5 · Evidence 52.7 · Response 52.6</details> | 56.5 | 47.2 | 50.2 | 52.0 |
| 16 | <details><summary>GPT-5 mini</summary><i>Perception:</i> Attribute 79.9 · Recognition 54.5 · Change 67.9 · Count 33.9<br><i>Reasoning:</i> Causality 60.1 · Spatial 45.6 · Temporal 41.1<br><i>Insight:</i> Intent 45.5 · Evidence 38.2 · Response 42.7</details> | 59.2 | 49.1 | 42.1 | 51.8 |
| 17 | <details><summary>GPT-4.1</summary><i>Perception:</i> Attribute 75.5 · Recognition 53.7 · Change 63.9 · Count 33.9<br><i>Reasoning:</i> Causality 61.7 · Spatial 42.6 · Temporal 35.2<br><i>Insight:</i> Intent 48.7 · Evidence 48.9 · Response 50.9</details> | 56.9 | 46.7 | 49.4 | 51.8 |
| 18 | <details><summary>Qwen3.8-Flash-Next</summary><i>Perception:</i> Attribute 73.5 · Recognition 43.1 · Change 71.5 · Count 35.1<br><i>Reasoning:</i> Causality 65.7 · Spatial 45.1 · Temporal 38.1<br><i>Insight:</i> Intent 50.8 · Evidence 45.7 · Response 42.7</details> | 55.9 | 49.9 | 46.5 | 51.7 |
| 19 | <details><summary>Qwen3.8-27B</summary><i>Perception:</i> Attribute 69.9 · Recognition 37.0 · Change 72.3 · Count 32.2<br><i>Reasoning:</i> Causality 66.1 · Spatial 38.4 · Temporal 42.8<br><i>Insight:</i> Intent 50.3 · Evidence 48.9 · Response 49.1</details> | 53.0 | 49.4 | 49.4 | 51.0 |
| 20 | <details><summary>Gemini 2.5 Flash</summary><i>Perception:</i> Attribute 81.5 · Recognition 52.0 · Change 70.3 · Count 29.0<br><i>Reasoning:</i> Causality 59.3 · Spatial 27.0 · Temporal 27.5<br><i>Insight:</i> Intent 56.7 · Evidence 48.9 · Response 57.3</details> | 58.3 | 38.3 | 54.2 | 50.9 |
| 21 | <details><summary>Qwen3.5-122B-A10B</summary><i>Perception:</i> Attribute 75.5 · Recognition 41.9 · Change 73.9 · Count 36.3<br><i>Reasoning:</i> Causality 59.3 · Spatial 39.7 · Temporal 39.0<br><i>Insight:</i> Intent 42.2 · Evidence 45.7 · Response 46.2</details> | 57.0 | 46.2 | 44.7 | 50.6 |
| 22 | <details><summary>Qwen3.6-35B-A3B</summary><i>Perception:</i> Attribute 75.1 · Recognition 38.6 · Change 69.9 · Count 36.7<br><i>Reasoning:</i> Causality 61.3 · Spatial 40.1 · Temporal 37.3<br><i>Insight:</i> Intent 44.4 · Evidence 41.4 · Response 48.5</details> | 55.2 | 46.5 | 44.7 | 49.9 |
| 23 | <details><summary>Qwen3.5-27B</summary><i>Perception:</i> Attribute 70.3 · Recognition 40.2 · Change 68.7 · Count 34.3<br><i>Reasoning:</i> Causality 62.9 · Spatial 38.8 · Temporal 34.3<br><i>Insight:</i> Intent 44.4 · Evidence 48.9 · Response 44.4</details> | 53.5 | 45.6 | 46.0 | 49.2 |
| 24 | <details><summary>Gemini 2.5 Flash-Lite</summary><i>Perception:</i> Attribute 76.7 · Recognition 51.2 · Change 57.4 · Count 36.7<br><i>Reasoning:</i> Causality 60.9 · Spatial 32.9 · Temporal 30.1<br><i>Insight:</i> Intent 43.9 · Evidence 36.6 · Response 50.9</details> | 55.6 | 41.6 | 43.6 | 48.2 |
| 25 | <details><summary>Qwen3.5-9B</summary><i>Perception:</i> Attribute 69.1 · Recognition 35.8 · Change 66.7 · Count 35.1<br><i>Reasoning:</i> Causality 58.9 · Spatial 38.0 · Temporal 38.6<br><i>Insight:</i> Intent 39.6 · Evidence 46.2 · Response 46.2</details> | 51.8 | 45.4 | 43.9 | 47.8 |
| 26 | <details><summary>MiMo-V2.6-Pro-MOPD</summary><i>Perception:</i> Attribute 66.7 · Recognition 32.5 · Change 67.5 · Count 35.1<br><i>Reasoning:</i> Causality 55.6 · Spatial 35.4 · Temporal 33.5<br><i>Insight:</i> Intent 48.1 · Evidence 47.8 · Response 41.5</details> | 50.6 | 41.7 | 46.0 | 46.6 |
| 27 | <details><summary>Qwen3-VL-8B Instruct</summary><i>Perception:</i> Attribute 66.3 · Recognition 37.0 · Change 61.5 · Count 27.4<br><i>Reasoning:</i> Causality 55.2 · Spatial 32.9 · Temporal 30.5<br><i>Insight:</i> Intent 48.7 · Evidence 45.7 · Response 62.0</details> | 48.2 | 39.8 | 51.8 | 46.4 |
| 28 | <details><summary>MiMo-V2.6-Pro-RL</summary><i>Perception:</i> Attribute 67.1 · Recognition 32.5 · Change 64.7 · Count 36.3<br><i>Reasoning:</i> Causality 55.6 · Spatial 36.3 · Temporal 31.8<br><i>Insight:</i> Intent 46.5 · Evidence 45.2 · Response 40.4</details> | 50.3 | 41.5 | 44.1 | 46.0 |
| 29 | <details><summary>GPT-4.1 mini</summary><i>Perception:</i> Attribute 73.1 · Recognition 49.2 · Change 50.2 · Count 27.8<br><i>Reasoning:</i> Causality 58.5 · Spatial 39.7 · Temporal 34.3<br><i>Insight:</i> Intent 40.6 · Evidence 36.6 · Response 41.5</details> | 50.2 | 44.4 | 39.5 | 45.8 |
| 30 | <details><summary>MiMo-V2.6-Flash-MOPD</summary><i>Perception:</i> Attribute 66.3 · Recognition 34.6 · Change 55.4 · Count 32.7<br><i>Reasoning:</i> Causality 55.2 · Spatial 39.7 · Temporal 26.3<br><i>Insight:</i> Intent 41.7 · Evidence 39.2 · Response 35.7</details> | 47.3 | 40.6 | 39.0 | 43.2 |
| 31 | <details><summary>MiMo-V2.6-Flash-RL</summary><i>Perception:</i> Attribute 67.9 · Recognition 34.1 · Change 56.6 · Count 33.1<br><i>Reasoning:</i> Causality 56.9 · Spatial 35.9 · Temporal 25.4<br><i>Insight:</i> Intent 43.3 · Evidence 34.9 · Response 36.3</details> | 48.0 | 39.7 | 38.2 | 43.0 |
| 32 | <details><summary>MiMo-V2.6-Distill-Qwen-9B</summary><i>Perception:</i> Attribute 63.5 · Recognition 29.7 · Change 59.8 · Count 32.7<br><i>Reasoning:</i> Causality 48.8 · Spatial 36.3 · Temporal 35.2<br><i>Insight:</i> Intent 28.9 · Evidence 38.2 · Response 35.1</details> | 46.5 | 40.2 | 34.0 | 41.5 |


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

## Citation

```bibtex
@misc{wang2026openhomeperceptioninsightsmarthome,
      title={OpenHome: From Perception to Insight in Smart-Home Camera Videos},
      author={Xianglong Wang and Neil Rodrigues and Pei Guo and Yiru Fang and Kevin Beussman and Shuyang Wang and Jingzhi Wang and Liangju Li and Shunyao Xu and Zhongwei Cheng and Wei Li and Lin Chen},
      year={2026},
      eprint={2610.31986},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.31986},
}
```

## License

[CC BY-NC 4.0](LICENSE)
