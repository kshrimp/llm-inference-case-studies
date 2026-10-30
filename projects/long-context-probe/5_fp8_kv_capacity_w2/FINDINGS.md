# Experiment 5 findings: FP8 KV on the W2 offline throughput benchmark

**Update (capacity-matched control, evening): most of the gain comes from cheaper attention, not capacity.** With the FP8 pool capped at BF16's 172,544 tokens, batching is identical to BF16 (decode batch max 119, 1,940 steps) and throughput is still **+47.6%** (1,813 → 2,676 tok/s). The doubled capacity adds a further **+10.5%** (→ 2,956 tok/s). See [the control](#capacity-matched-control-splits-the-gain).

**Result: storing KV in FP8 (saturating v2 patch) raised mini-SGLang's offline throughput on the sprint's W2 benchmark by 62.6%, from 1,818 to 2,956 tok/s (two runs each, order BF16, FP8, FP8, BF16).** The KV pool's token capacity doubled (172,544 → 345,344), the scheduler admitted twice as many concurrent requests (decode batch max 119 → 238), and the number of decode steps fell from 1,940 to 1,182. This is **offline throughput** for one workload on one L4, not serving latency.

Reproduce every number below with `uv run python projects/long-context-probe/5_fp8_kv_capacity_w2/table.py`.

## Results

| Order | Variant | Pool tokens | Generated / requested | Time (s) | Throughput (tok/s) | Decode steps | Decode batch max / mean | Run ID |
| ---: | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | BF16 | 172,544 | 133,966 / 133,966 | 73.787 | 1,815.57 | 1,940 | 119 / 68.92 | `20261024T052306Z-bench-bf16-r1` |
| 2 | FP8 (sat) | 345,344 | 133,966 / 133,966 | 45.638 | 2,935.40 | 1,182 | 238 / 113.12 | `20261024T052654Z-bench-fp8_sat-r1` |
| 3 | FP8 (sat) | 345,344 | 133,966 / 133,966 | 44.996 | 2,977.27 | 1,182 | 238 / 113.12 | `20261024T052817Z-bench-fp8_sat-r2` |
| 4 | BF16 | 172,544 | 133,966 / 133,966 | 73.578 | 1,820.75 | 1,940 | 119 / 68.92 | `20261024T052940Z-bench-bf16-r2` |

- **Throughput:** BF16 mean 1,818.16 tok/s (runs 5.2 tok/s apart); FP8 mean 2,956.34 tok/s (41.9 tok/s apart). **+62.6%**. The larger repeat spread is under 4% of the difference.
- **Baseline consistency:** the BF16 control is within 1% of the W2 baseline recorded on (1,798.23 and 1,807.57 tok/s) under the older dependency set, so the newly pinned dependencies did not shift the baseline materially.
- **Mechanism:** the first decode batches were 119 requests in BF16 and 238 in FP8, exactly 2×. The scheduler is deterministic: the step counts and batch statistics are identical across repeats. With more requests per step, each decode step amortizes the per-step weight reads (GEMV) over more tokens, so the run finishes in 39% fewer steps.

## Capacity-matched control splits the gain

`--variant fp8_sat_capmatch` runs the same FP8 image with `num_page_override=674`, the BF16 page count, so the pool holds exactly BF16's 172,544 tokens. It ran in the order BF16, capmatch, capmatch, BF16.

| Order | Variant | Pool tokens | Time (s) | Throughput (tok/s) | Decode steps | Decode batch max / mean | Run ID |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | BF16 | 172,544 | 73.691 | 1,817.93 | 1,940 | 119 / 68.92 | `20261024T183054Z-bench-bf16-r3` |
| 2 | FP8, capacity-matched | 172,544 | 50.098 | 2,674.09 | 1,940 | 119 / 68.92 | `20261024T183333Z-bench-fp8_sat_capmatch-r1` |
| 3 | FP8, capacity-matched | 172,544 | 50.008 | 2,678.89 | 1,940 | 119 / 68.92 | `20261024T183502Z-bench-fp8_sat_capmatch-r2` |
| 4 | BF16 | 172,544 | 74.085 | 1,808.27 | 1,940 | 119 / 68.92 | `20261024T183630Z-bench-bf16-r4` |

All 133,966 / 133,966 requested tokens were generated in every run.

- **Same batching, faster steps:** capacity-matched FP8 has exactly BF16's schedule (1,940 steps, batch max 119, mean 68.92), yet it is **+47.6%** faster (mean 2,676.49 vs 1,813.10 tok/s), with mean decode-step time down 32%. This share comes from halving KV bytes per attention read.
- **Capacity adds the rest:** full-capacity FP8 (2,956.34 tok/s, first table) is **+10.5%** above capacity-matched FP8. This share comes from 2× concurrency and 39% fewer steps.
- **Comparison across sessions:** all four BF16 runs (two in each session) fall between 1,808.27 and 1,820.75 tok/s, a 0.7% spread, so comparing the morning and evening runs is reasonable.
- **Why attention dominates here, unlike the single-request case** (estimate, not measured): at about 69 concurrent requests with contexts of hundreds to about 2,000 tokens, a decode step's BF16 KV reads (114,688 bytes per token) are several GB, against about 1.2 GB of weights per step. Halving KV bytes therefore shortens most of each step. In experiment 3's single 8,000-token request, attention was only about 40% of step time, giving −13.7%. A batch-size × context sweep with traces would confirm this split.

## What this does and does not show

- The capacity-matched control separates the two factors: about +47.6% from cheaper attention and about +10.5% from more concurrency (multiplicative, on this workload).
- **Accuracy:** on 200 held-out arithmetic problems through the chat template (greedy), FP8 v2 scored **180/200 (90.0%)** against BF16's **182/200 (91.0%)**. Their text was identical on 174 problems, and the 4-vs-2 correctness disagreement is within noise (exact McNemar p ≈ 0.69). FP8 v1 scored 1/200, with every output corrupted. See [experiment 3 findings](../3_fp8_kv_cache_feasibility/FINDINGS.md#accuracy-eval-). Quality under this benchmark's temperature-0.6 sampling was not measured directly.
- This is a single model, GPU, and workload, offline only. Client-observed serving latency and time-to-first-token were not measured.

## Next steps

1. ~~Capacity-matched FP8 control.~~ Done: +47.6% from attention and +10.5% from capacity.
2. ~~Small accuracy evaluation.~~ Done: 90.0% vs 91.0% on 200 arithmetic problems, within noise. A broader evaluation (more tasks, longer contexts, logprob comparison) remains before any production recommendation.
3. Optional: a batch-size × context sweep with traces to confirm the attention-dominance estimate.
4. Fusing the clamp and cast into the store kernel would recover about 0.18 ms per step out of roughly 8 ms in the single-request case. That is small next to this result, and it needs a custom CUDA kernel, which the project instructions exclude without a recorded reason. Deferred.
