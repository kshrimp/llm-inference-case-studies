# Experiment 10: prefill/decode interference in mini-SGLang (offline)

**Question.** When long prompts are admitted while short requests are decoding, how long do the short requests stall, and does the prefill chunk size (`max_extend_tokens`) change that? This is an offline, single-engine version of a serving concern: a long prefill can delay already-running decodes even when aggregate tokens/s looks fine.

## Why we ran this

Experiments 3–9 measured throughput and correctness. A serving customer, however, usually complains about the *tail*: chat users whose stream freezes when someone else pastes a long document. Aggregate tokens/s can look healthy while that happens. Reading the scheduler suggested that mini-SGLang would stall running decodes for the whole long prefill, and that the usual fix, chunked prefill, would not help without a decode-aware policy. Measuring this offline first gives a known mechanism and expected magnitude to look for in the serving case study's online version (S2), and it maps directly onto the scheduler section of the mini-SGLang architecture map.

## What we did

1. Read `Scheduler._schedule_next_batch` and confirmed the prefill-first policy (explained below).
2. Added an observe-only wrapper that logs every scheduled step's start time, kind, request ids, and extend tokens.
3. Ran five configurations in one container, each in a fresh engine subprocess: 32 short requests alone, with 1 long prompt, and with 4 long prompts at three chunk sizes. A second container repeated all five.
4. Computed each configuration's stall (the short requests' largest gap between steps), median gap, completion time, and the long prompts' first-token time with [`table.py`](table.py).

**Why mini-SGLang should stall (source, pinned `9a91cfaf`).** `Scheduler._schedule_next_batch` runs `prefill_manager.schedule_next_batch(...) or decode_manager.schedule_next_batch()`, which is strictly **prefill first**; the code has a `TODO: support other policies: e.g. DECODE first`. While any pending prefill fits the budget and pool, no decode step runs. Chunking splits a long prefill into more steps but does not insert decode steps between them. Changing the policy would be a custom scheduler, which the project instructions exclude, so this experiment only measures.

**Design.** The runner is `--stage interference` in [experiment 3's app.py](../3_fp8_kv_cache_feasibility/app.py), BF16 with the unmodified image. One container runs each configuration in its own subprocess, so each gets a fresh engine. A repeat container follows.

| Config | Short requests (128 in / 256 out, greedy) | Long requests (8,000 in / 16 out) | `max_extend_tokens` (prefill chunk) |
| --- | ---: | ---: | ---: |
| control | 32 | 0 | 8,192 |
| 1long_chunk8k | 32 | 1 | 8,192 |
| 4long_chunk2k | 32 | 4 | 2,048 |
| 4long_chunk8k | 32 | 4 | 8,192 |
| 4long_chunk16k | 32 | 4 | 16,384 |

- Short requests are listed first, so they prefill first and are decoding when the long prompts' prefill is scheduled.
- A wrapper on `_schedule_next_batch` logs every step's start time, kind (prefill/decode), request ids, and extend tokens. It only observes.
- **Per config:** the short requests' **largest gap between consecutive steps** (the stall), median and p99 gap, median/max completion time, median time to their first decode step, the long requests' first-token time, total elapsed, and the step log.
- Times are step **start** timestamps under the overlap scheduler, an approximation of token times, offline. This is not client latency.

**Expected signal (agent-written).** The control has no stall. With long prompts, the short requests' largest gap is about the total long-prefill time. Experiment 2 measured about 0.30 s from first GPU event to first decode for one 8,000-token prompt, so expect about 0.3 s with 1 long prompt and about 1.2 s with 4. The stall grows with the number of long prompts and is roughly independent of chunk size, because prefill-first runs all chunks back to back.

## How to run

Run from the repository root. `bash projects/long-context-probe/10_prefill_decode_interference/run.sh` runs everything below in order.

```bash
cd /path/to/llm-inference-case-studies
A=projects/long-context-probe/3_fp8_kv_cache_feasibility/app.py
uv run modal run $A::main --stage interference --variant bf16 --repeat 1
uv run modal run $A::main --stage interference --variant bf16 --repeat 2
uv run python projects/long-context-probe/10_prefill_decode_interference/table.py
uv run modal app list
```

Cost: 2 containers with a 30-minute timeout, $0.58 worst case each. Results go to `results/long-context-probe/minisgl-prefill-decode-interference/`.

## Files

- [`run.sh`](run.sh), [`table.py`](table.py): commands; per-config stall and latency table across repeats.
- Results and analysis: [FINDINGS.md](FINDINGS.md).
