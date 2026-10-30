# LLM inference case studies

Reproducible studies of inference performance and correctness on Modal. These are personal engineering projects, not upstream SGLang contributions.

| Study | Question | Main measured result |
| --- | --- | --- |
| [Qwen3-Omni on Modal](projects/qwen3-omni-on-modal/README.md) | How much 8k-input / 1k-output text traffic can one H100 serve within a latency target? | Tuned FP8: 1.357 requests/s, 2.8× tuned BF16, under pre-set p95 first-token ≤3 s and time-between-tokens ≤60 ms targets. |
| [FP8 KV cache in mini-SGLang](projects/long-context-probe/README.md) | When do lower KV bandwidth and extra capacity help? | 62.6% higher offline throughput on an L4; 47.6% at equal KV capacity. The same study includes a locally reproduced prefix-cache bug (84/90 → 90/90 retrieval). |
| [Qwen3-Omni audio and speech](projects/omni-pipeline-probe/README.md) | Where does audio latency go, and can speech start earlier? | CPU preprocessing explained 45.5 ms of a 65.9 ms first-text increase from 10 to 90 s clips; partial talker start cut first audio from ~340 to ~226 ms for short replies. |

## How to use this repo

Each study contains its question, pinned engine and model revisions, commands, validation rules, findings, and limits. Read the study README before running any command: GPU runs use paid Modal resources and expect your own Modal account and model access. The Python environment is pinned in `pyproject.toml` and `uv.lock`; start with `uv sync` and the CPU checks in each study.

Only code, study notes, and summarized findings are published here. Raw traces, request outputs, model weights, billing records, local planning files, and credentials stay outside this repo. Published figures are reported from the original validated runs; the raw artifacts are not included, so the figures cannot be independently recalculated from this checkout. The commands and validation code show how to collect a fresh run.

The FP8 Qwen3-Omni weights are a [community conversion](https://huggingface.co/marksverdhei/Qwen3-Omni-30B-A3B-FP8). The capacity and cold-start conclusions apply to the documented text-only workload and one H100. Offline throughput, client-observed latency, and in-engine measurements are reported separately.
