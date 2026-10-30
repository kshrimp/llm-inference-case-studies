# Experiment 10 findings: long prefills freeze in-flight decodes, and chunking does not help (2026-10-25)

**Result: under mini-SGLang's prefill-first scheduler, four 8,000-token prompts stall all 32 decoding short requests for about 1.2 s**, against a normal 10.4 ms decode step. The short requests finish 55% later, while their **median** time between tokens is unchanged. The prefill chunk size does not relieve the stall: 8k and 16k chunks are equivalent, and 2k chunks are about 0.1 s worse because of their extra steps. This matches the source: `_schedule_next_batch` tries prefill before decode, so chunks run back to back. Offline, single engine, one L4, BF16.

Reproduce with `uv run python projects/long-context-probe/10_prefill_decode_interference/table.py`. Raw records are in `results/long-context-probe/minisgl-prefill-decode-interference/`; each config includes its step log.

## Runs

There are two repeats, one container each: `20261025T042653Z-interference-r1` and `20261025T042845Z-interference-r2`. A first pair (`20261025T042001Z`, `20261025T042207Z`, kept in `failed/`) failed in every worker subprocess before any measurement, because of patch hashing on re-import; that is fixed in the runner.

| Config | Worst gap for short requests (ms) | Median gap (ms) | Short requests' first decode step (s) | Short completion (s) | Long first token (s) | Prefill steps / all steps | Elapsed (s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| control (32 short) | 85.7 / 86.6 | 10.19 / 10.20 | 0.091 / 0.090 | 2.656 / 2.674 | — | 1 / 256 | 2.682 / 2.701 |
| + 1 long, chunk 8,192 | 236.8 / 242.4 | 10.43 / 10.40 | 0.243 / 0.247 | 3.164 / 3.067 | 0.243 / 0.247 | 2 / 257 | 3.190 / 3.092 |
| + 4 long, chunk 2,048 | **1,290.2 / 1,317.8** | 10.37 / 10.47 | 1.294 / 1.320 | 4.145 / 4.192 | 1.294 / 1.320 | 18 / 273 | 4.171 / 4.218 |
| + 4 long, chunk 8,192 | 1,168.2 / 1,195.5 | 10.36 / 10.44 | 1.175 / 1.200 | 4.119 / 4.164 | 1.175 / 1.200 | 5 / 260 | 4.146 / 4.190 |
| + 4 long, chunk 16,384 | 1,179.5 / 1,207.4 | 10.40 / 10.47 | 1.187 / 1.213 | 4.128 / 4.176 | 1.187 / 1.213 | 3 / 258 | 4.155 / 4.201 |

Values are r1 / r2. Short requests: 128 input / 256 output tokens; long requests: 8,000 input / 16 output tokens, all greedy. "Gap" is the time between consecutive scheduled steps that include a given short request.

## Analysis

- **What the stall is:** the short requests' first token comes from their own prefill (about 0.09 s). Their next decode step waits until **all** pending long-prompt prefill has run, so their first-decode time equals the long requests' first-token time in every config. The worst gap is that wait.
- **Scaling:** the stall is about 0.15 s with 1 long prompt (0.24 s minus the control's 0.09 s) and about 1.1 s with 4, roughly linear in the long prefill work. This matches experiment 2's measurement of about 0.30 s of pre-decode GPU work for one 8,000-token prompt.
- **Chunk size:** with prefill-first scheduling, smaller chunks only add steps. 2,048-token chunks take 18 prefill steps and add about 0.11 s over 8,192 or 16,384, which are within about 1% of each other. Chunking can only help if the scheduler interleaves decode steps between chunks (a mixed or decode-aware policy). The pinned scheduler does not; its source carries `TODO: support other policies: e.g. DECODE first`.
- **Averages hide it:** the median gap moves only from 10.2 to 10.4 ms, and total elapsed grows 55% (2.68 → 4.15 s). A per-request worst-gap or p99 time-between-tokens metric is needed to see interference. One 1-long-prompt repeat shows a p99 of 161 ms vs 28 ms in the other; with only one stall gap per short request, p99 is noise-sensitive here, so the worst gap is the robust statistic.

## What this means

- For a serving deployment of this engine, one long prompt arriving while short chats are streaming produces a visible freeze for every active user, about 0.3 s per 8,000 prompt tokens on an L4. Throughput dashboards would not show it.
- Mitigations exist in full engines, such as decode-aware or mixed chunked prefill and limits on prefill tokens per step. Measuring them is the online version of this experiment, planned as serving-track **S2** (see the [study index](../README.md#follow-up-options)). Implementing a new policy in mini-SGLang would be a custom scheduler, which is out of scope without a recorded reason.

## Limits

- Offline, single engine; all requests are submitted at once, with short requests listed first. This is not a real arrival process, and times are step **start** timestamps under the overlap scheduler, not client-observed latency.
- One model and GPU, random-token prompts, BF16, radix cache on (the prompts share no prefixes).

## Cost

Four containers in total (the failed pair plus two good runs); actuals are on the study-plan status board.
