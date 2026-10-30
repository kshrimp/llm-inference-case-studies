# Serving Qwen3-Omni on Modal for long text inputs: recommendation

**For:** a team planning to serve Qwen3-Omni-30B-A3B on Modal for requests of about 8,000 input and 1,000 output tokens (text in, text out).
**Basis:** measurements on one Modal H100 (80 GB) with SGLang-Omni `305273a6`. Every number below links to the experiment that produced it.

## Recommendation in one paragraph

Run the **FP8 checkpoint with two backend flags and a raised memory fraction**. On one H100 that serves **about 1.36 requests per second (about 4,900 per hour) at p95 time to first token under 3 s**. That is 2.8× tuned BF16 and far beyond the out-of-the-box setup, which fails the target at 0.21 requests per second. Estimated cost: **about $0.99 per 1,000 requests** at Modal's list price when the GPU is kept busy. Plan for cold starts of **3 to 5 minutes**. Keep compiled kernels on a Volume, which cut the median cold start by 31% in our runs (275 → 190 s). Keep at least one warm container whenever a first token within seconds matters: a cold start cannot get there on this stack today, and Modal's GPU memory snapshots failed for this server.

## The configuration

```bash
sgl-omni serve --model-path marksverdhei/Qwen3-Omni-30B-A3B-FP8 --text-only \
 --thinker.engine.mem_fraction_static 0.90 \
 --thinker.engine.fp8_gemm_runner_backend deep_gemm \
 --thinker.engine.moe_runner_backend triton
```

| Setting | Why |
| --- | --- |
| FP8 weights | Half the weight bytes: a 424k-token KV pool instead of 121k at the same memory setting ([4a](4_capacity/FINDINGS.md)). **This is a community FP8 conversion, not an official Qwen release.** It matched BF16 on 60/60 needle-retrieval items and on the first 100 greedy tokens of an 8k prompt ([2](2_fp8_thinker/FINDINGS.md)). That rules out a large quality loss, not a small one, so run your own evaluation before production |
| `fp8_gemm_runner_backend deep_gemm` + `moe_runner_backend triton` | Without them, the automatic FP8 backends decode **46% slower than BF16** (6.52 vs 4.46 ms/token). With them, FP8 is **7.6% faster than BF16** (4.12 ms/token) ([2b](2_fp8_thinker/FINDINGS.md)) |
| `mem_fraction_static 0.90` | The automatic value leaves 14 GiB of the GPU unused and fits only 2 concurrent 8k requests. 0.90 peaked at 92.4% of GPU memory in every load test; 0.93 went over a 95% safety line ([4a](4_capacity/FINDINGS.md)) |

**If you cannot accept DeepGEMM's post-start compiles**, use `fp8_gemm_runner_backend triton` with tuned tiles for the two dense shapes. That decodes at 4.39 ms/token (1.6% faster than BF16, 6.5% slower than DeepGEMM) ([2c](2_fp8_thinker/FINDINGS.md)). SGLang does not ship those tiles yet; we generated them with SGLang's own tuning function.

## How much traffic one H100 carries

Open-loop Poisson arrivals of 8k/1k requests, with targets set before the runs: p95 time to first token ≤ 3 s and p95 time between tokens ≤ 60 ms ([4b](4_capacity/FINDINGS.md)).

| Configuration | Highest rate that met the target | Requests per hour | Est. $ per 1,000 requests |
| --- | ---: | ---: | ---: |
| **FP8, recommended flags, 0.90** | **1.357 req/s** | **≈ 4,890** | **≈ $0.99** |
| BF16, 0.90 | 0.490 req/s | ≈ 1,760 | ≈ $2.74 |
| BF16, defaults | none at ≥ 0.21 req/s | < 760 | > $6.40 |

**Sizing rule:** about 1.3 requests per second per H100 for this request shape. **The limit is queueing, not generation speed.** Per-token latency stayed at or below 27 ms at p95 at every tested rate, while p95 time to first token jumped from 0.3 s to over 2.5 s at the limit. Scale out before a GPU reaches about 1.15 requests per second: at that rate p95 time to first token was still 0.33 s, and one step higher it was 2.55 s. Each new 8k prompt pauses running streams for about 150 ms while it is processed. Users of a streaming UI see this as an occasional stall, at p99 rather than p95.

The dollar figures are list price ÷ throughput. They exclude idle time and cold starts, which in practice dominate at low traffic.

## Cold starts

From a scaled-to-zero app to the first token of a short request, one H100, client-observed ([3](3_cold_start/FINDINGS.md)):

| Setup | Cold start, client-observed (runs) | Largest phases inside the container |
| --- | --- | --- |
| FP8 recommended, fresh container | 201, 275, 278 s (median 275) | Prefill CUDA-graph capture with DeepGEMM compiles 101–144 s; spawn and imports 54–82 s; weights 21–61 s |
| **FP8 recommended, kernel caches on a Modal Volume** | **174, 190, 199 s (median 190)** | Capture 60–70 s; spawn and imports 60–71 s; weights 22–27 s |
| BF16, fresh container | 165, 238 s | Weights 55–79 s; spawn and imports 56–88 s; capture 28–34 s |
| FP8 with a GPU memory snapshot | **Failed**: snapshot creation errored twice | SGLang-Omni's multi-process GPU server is not snapshot-compatible as built |

What to do:
- **Persist the kernel caches:** mount a Volume and set `SGLANG_CACHE_DIR`, `TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE` and `TORCHINDUCTOR_CACHE_DIR` under it. Prime it once with a single container.
- **Size warm capacity for your latency promise:** with 3-minute cold starts, `min_containers` ≥ 1 is the only way to keep first-token latency in seconds. One warm H100 costs about $4.84 per hour, about $3,500 per month.
- **Expect spread.** Weight-load speed from the Volume varied 3× between containers (0.48–1.40 GB/s), as in every earlier experiment.

## What we did not test

Mixed request lengths, audio/image inputs, speech output, multi-GPU, H200/B200, quality beyond a 60-item retrieval set, and network latency to a Modal web endpoint (all clients ran inside the container). Each capacity result is one run per rate. The knee rates could move a step on a repeat.
