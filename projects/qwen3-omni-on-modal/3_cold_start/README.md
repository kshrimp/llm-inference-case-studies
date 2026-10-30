# Experiment 3: cold start, zero to first token (systems)

**Question.** When a Modal app serving Qwen3-Omni scales from zero, how long until the first token, which phases inside the container take that time, and which platform levers cut it? This is the question a Modal customer with bursty traffic asks first, and cold starts are billed time.

## Clocks: what is observed, and what is not subtracted

Timestamps from different clocks are never subtracted from each other.

- **Client span (client clock, `time.monotonic`)**: request sent → first streamed token received. This is the customer-visible cold start.
- **Container spans (container clock, `time.monotonic` in the entrypoint process; stage processes report `time.time()` and are converted using one offset measured at process start)**: entrypoint start → imports done → weights loaded → engine ready (stage processes report ready) → warmup done → first token sent. Each container phase is a difference of two container timestamps.
- **Not observed:** time before the entrypoint starts (Modal scheduling, image pull, container boot). It is reported as an **unobserved interval** whose size is bounded only by the client span minus the sum of container spans, and labelled as such, never as a measured phase. If Modal exposes container start events through the API or logs, they are recorded as a separate span on Modal's clock.

## Cache state recorded for every cold run

| State | How it is recorded |
| --- | --- |
| Container is new | `MODAL_TASK_ID` differs from every earlier run; previous container confirmed scaled to zero (`modal app list` / container list) before sending |
| Image layers on the worker | Not observable from inside; recorded as unknown |
| Weights | Read from the `omni` Volume every time; Volume revision recorded |
| JIT / FlashInfer / DeepGEMM / CUDA-graph caches | Directory hash and file count of each cache directory at start, and whether it is a fresh or populated Volume |
| Memory snapshot | none / created / restored, from Modal's snapshot flags and the entrypoint's restore hook |

## What experiment 1 already shows (not controlled cold starts)

Server-log phases from two containers ([experiment 1 review](../1_long_context_anatomy/FINDINGS.md#review-checks--no-new-gpu-runs)): about 39 s of process spawn and imports, **weight load 47–177 s for the same 56.9 GB (1.20 vs 0.32 GB/s)**, prefill CUDA-graph capture 30–34 s, decode graph capture 4–5 s, and a 14–18 s warmup request. Server ready took 146–282 s. Two consequences:

- **Under 30 s is not reachable by tuning the normal start.** Loading the BF16 weights alone in 30 s needs at least 1.9 GB/s, faster than either observed rate, before any other phase. Only a restored memory snapshot can skip loading. The 30 s target therefore applies to the snapshot condition. For the non-snapshot conditions the target is **≤ 90 s**, which needs fast loading and the graph-capture lever.
- **Weight-load variance is itself a finding.** A 3.7× spread on the same Volume makes p95 cold start much worse than the median. Every condition records weight-load seconds and GB/s from the server log, and the report shows the spread, not only the median.

## Conditions (3 cold containers each)

| # | Condition | What changes | Risk |
| --- | --- | --- | --- |
| 1 | Baseline | Empty JIT-cache Volume each run | none |
| 2 | Startup-tuned normal start | Populated JIT/FlashInfer/Triton cache Volume, **and** fewer prefill CUDA-graph buckets (`--thinker.engine.cuda_graph_max_bs_prefill` lowered, or prefill graphs off), which removes most of the 30–34 s capture | Prefill graphs speed up prefill; the TTFT cost is measured on the same 8k prompt as experiment 1, against its 178 ms |
| 3 | Memory snapshot | Modal memory snapshot taken after engine ready (CPU snapshot; GPU snapshot if the platform option is available) | SGLang-Omni runs several stage processes, which may not restore. A failed restore is a valid finding |

FP8 weights as a cold-start lever (37.4 instead of 56.9 GB to read) is added as a fourth condition only if experiment 2 passes B and the budget allows it.

### Revised conditions (before any exp 3 run)

Experiments 2 and 4a changed what a customer would deploy: FP8 (community checkpoint) with `deep_gemm` dense + `triton` MoE at `mem_fraction_static` 0.90. Server logs from 2b-2, 4a and 4b show that this config's startup cost is not where experiment 1 put it:

| Fresh container, server log | Weight load | Prefill CUDA-graph capture (58 token buckets) | Server ready |
| --- | ---: | ---: | ---: |
| BF16 (exp 1, two containers) | 47–177 s (56.9 GB) | 30–34 s | 146–282 s |
| FP8 + DeepGEMM dense (2b-2, 4a, 4b) | 38–43 s (29.2 GB) | **101–127 s** | 186–265 s |
| FP8, Triton dense (2b-2 `B_default`, page cache warm, second server in its container) | 11.9 s | 21 s | 74 s |

The DeepGEMM capture time fits per-shape JIT compiles: SGLang-Omni sets `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`, and SGLang's DeepGEMM cache defaults to `~/.cache/sglang/deep_gemm` on the container disk, so every new container compiles again. So the conditions now measure the recommended FP8 config, with one lever at a time:

| # | Condition | Cache roots (`TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `TORCHINDUCTOR_CACHE_DIR`, `SGLANG_CACHE_DIR`) | Containers |
| --- | --- | --- | ---: |
| F1 | FP8 baseline | Fresh directory on the container disk (`/tmp/fresh-cache`) | 3 |
| F2 | FP8 startup-tuned: compiled kernels persisted | `/cache/warm-fp8` on the `qwen3-omni-jit-cache` Volume, filled by one unmeasured priming container | 1 + 3 |
| F3 | FP8 memory snapshot (GPU snapshot, alpha) | As F2 | 1 creation + 2 restores, stop at the first failure |
| B1 | BF16 baseline, for comparison | Fresh | 2 |

- **Prefill-graph buckets are not reduced.** Once the compile cache is persisted, the remaining capture should be about 15–21 s (the warm captures above), so a bucket change would save little and would cost TTFT. Keeping one lever per condition keeps the attribution clean.
- **The snapshot is an attempt with a known risk.** Modal's SGLang recipe releases GPU memory (`/release_memory_occupation`) before the snapshot. SGLang-Omni has no such endpoint for Qwen3-Omni at this pin, and Modal documents GPU snapshots as generally incompatible with multi-process GPU code; the server runs several stage processes linked by CUDA IPC. A failed restore is reported with its error.
- **Correctness gate:** each run generates 32 greedy tokens for experiment 1's S prompt (807 words, 1,150 tokens). A run is valid only if it returns exactly 32 tokens and its text is byte-identical to every other run of the same model. The earlier "exact match with experiment 1" gate cannot apply to FP8, and experiment 1 saved only 1,000-token hashes.
- **Runner:** a separate deployed App (`app.py::cold_app`, classes `ColdStart` and `ColdSnapshot`), because memory snapshots exist only for deployed Apps. The client `client.py` waits until the App has no container before each run. The client span runs from the call to the first streamed token (local clock). Container spans and server-log phases are on the container clock; their gap is reported as an upper bound on the unobserved time.

**Hard targets:** under 30 s client-observed for the snapshot condition; ≤ 90 s for the startup-tuned normal start. A measured miss with its container-phase breakdown is a valid result.

**Validity:** a run counts only if the container is new, the cache-state record is complete, and the first token is correct (exact match with experiment 1's first 32 tokens for the same prompt, short prompt used here). Each container is capped at 10 minutes of startup; hitting the cap is reported as a failed cold start, not retried.
