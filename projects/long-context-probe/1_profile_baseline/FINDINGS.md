# First long-context probe findings —

## What ran

All four runs used mini-SGLang configured at pin `9a91cfafe754aa85daee49998176275667eb58f2`, the existing Qwen3-0.6B model Volume, bf16, one L4, one synthetic 8,000-token input, and the settings in `app.py`. The recorded model revision is `c1899de289a04d12100db370d81485cdf75e47ca`, but this probe checks only that `config.json` exists; it does not independently hash the mounted snapshot. These are **offline** `LLM.generate` calls, not serving latency measurements. Raw `run.json` and Chrome traces are in the git-ignored `results/long-context-probe/` folders named below.

| Mode | Run ID suffix | Requested / actual output | Timed `generate` call |
| --- | --- | ---: | ---: |
| smoke | `24bf40f5-smoke` | 8 / 8 | 4.011 s |
| profile | `2b6cf5ae-profile` | 32 / 32 | 1.598 s with profiler |
| full | `a696d1a0-full` | 1,000 / 1,000 | **9.751 s without profiler** |
| full-profile | `e98241a4-full-profile` | 1,000 / 1,000 | 13.892 s with profiler |

The profile and full runs are separate containers, so their elapsed-time difference is **not** an isolated estimate of profiler overhead. The short smoke was slower than the 32-token run; these first runs do not establish why. No repeat of the unprofiled full run exists yet. Each result's `valid_count` is true, which checks requested against returned output tokens.

## What the full trace shows

Run `uv run python projects/long-context-probe/1_profile_baseline/analyze_trace.py results/long-context-probe/20261023T180927Z-e98241a4-full-profile/trace.json` to reproduce the event grouping below. The trace file is about 382 MB. The script streams events, so it does not load the entire file into memory.

| Recorded event group | Count | Sum of event durations |
| --- | ---: | ---: |
| FlashInfer paged-KV **decode attention** kernels | 27,972 | 3,940.8 ms |
| cuBLAS matrix-vector (`gemvx`) kernels | 112,888 | 5,041.8 ms |
| FlashInfer paged-KV **prefill attention** kernels | 28 | 135.2 ms |
| Other GPU kernels | 234,132 | 745.9 ms |
| CPU `cudaEventSynchronize` calls | 2,000 | 8,006.8 ms |

GPU kernel event sums are diagnostic, not wall-time partitions: different streams and categories can overlap. The CPU synchronization time overlaps GPU work and should not be called CPU computation. The trace has no explicit prefill/decode phase markers, so the table cannot give complete phase elapsed times. Paged-KV decode attention and matrix-vector kernels dominate the recorded GPU kernel durations for this single-request workload. The 28 prefill-attention events correspond to attention work at the start of the request, but other prefill kernels are grouped elsewhere.

Pinned source pointers: `python/minisgl/llm/llm.py` (`LLM.generate`), `python/minisgl/scheduler/scheduler.py` (batch selection and result processing), and `python/minisgl/attention/fi.py` (FlashInfer prefill/decode wrappers) in the external mini-SGLang clone. Match the trace's kernel names to these call paths before attributing a kernel to an engine-level decision.

### Decode-step trend visible in the existing trace

Run `uv run python projects/long-context-probe/1_profile_baseline/decode_step_trend.py results/long-context-probe/20261023T180927Z-e98241a4-full-profile/trace.json` to reproduce this view. It sorts the 27,972 decode-attention kernel events by GPU timestamp and groups **28 kernels per inferred step**, matching the model's 28 layers and yielding 999 decode steps. The main cuBLAS GEMV kernel appears 112 times per inferred step. These are inferred groups, not explicit scheduler markers.

| Per-step trace measure | First 100 steps, median | Last 100 steps, median | Change |
| --- | ---: | ---: | ---: |
| Sum of 28 decode-attention kernel durations | 3.742 ms | 4.149 ms | +10.88% |
| Interval between starts of successive attention groups | 9.322 ms | 9.755 ms | +4.64% |
| Sum of main matrix-vector kernel durations | 3.771 ms | 3.780 ms | +0.23% |

As output grows from the start toward 1,000 tokens, the attention work increases while this matrix-vector work stays nearly flat. That is consistent with growing KV context affecting decode attention. It is **one profiled request**, and the trace does not measure memory bandwidth or prove FP8 would help. Kernel sums can overlap; the start interval includes other work. The [context-length validation](../2_context_length_validation/README.md) now pairs unprofiled whole-call repeats with three matched short traces to test this pattern at different initial lengths.

## Hypothesis and next validation

**Hypothesis:** For this single 8,000/1,000 request, decode-attention cost rises as the KV context grows; matrix-vector work is substantial but relatively flat per step. The trace shows large cumulative durations for both over 1,000 output steps. **Alternative:** the observed trend may depend on this graph/backend, clock or profiling effects, or another operation correlated with context length; event sums do not establish a fix or achievable speedup.

The next *scoped* test is specified in the context-length validation runbook: compare a few fixed input lengths with two unprofiled repeats at 1,000 output tokens and three matched short traces at 128 output tokens. Rising decode-attention kernel time with input length would support investigating KV access; a flat trend would weaken that case. If more precise phase attribution is needed, add explicit markers in the external clone. Before an FP8 KV proposal, measure pool bytes/capacity and validate output quality. Before kernel fusion, identify specific launches or intermediate tensors that consume material time. No engine optimization or upstream issue is claimed from this trace alone.

## Cost and cleanup handoff

The four local `run.json` files record wall-clock start/end and timed `generate` duration, **not Modal billable seconds**. The user supplied a `modal billing report --start "$START_DATE" --end "$END_DATE" --csv` export . Its four `long-context-probe` rows total **$0.11884565** for the four modes combined; the report groups charges by app, so it does not support a per-mode bill. The user reports all apps closed. This cleanup state was user-confirmed, not independently observed by this agent because the Modal CLI listing did not return here. The downloaded traces have been moved from the repository root into their matching git-ignored result folders. No additional GPU run was launched for this analysis.
