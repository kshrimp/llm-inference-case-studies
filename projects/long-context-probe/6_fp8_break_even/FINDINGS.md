# Experiment 6 findings: FP8 KV break-even map

**Result: FP8's decode-step saving is set by the KV bytes each step reads.** The gain runs from neutral or −1% for one 256-token request to +49% throughput for 32 × 4,096 tokens. A two-parameter bandwidth model fits all 16 cells (R² = 1.000), with break-even at about 616 tokens in flight and a 45% step-time ceiling.

Reproduce with `uv run python projects/long-context-probe/6_fp8_break_even/table.py`.

## Runs and per-cell results

All 4 containers completed (BF16 `20261025T034350Z`, FP8 `20261025T034549Z`, FP8 `20261025T034719Z`, BF16 `20261025T034900Z`). Every cell generated batch × 256 tokens, and the **decode batch max equaled the requested batch in every cell**, so pool size never limited admission. Raw records are in `results/long-context-probe/fp8-kv-break-even/`.

| Context | Batch | BF16 KV in flight (GB, end) | BF16 tok/s (r1, r2) | FP8 tok/s (r1, r2) | Throughput change | Decode step BF16 → FP8 (ms) | Step change |
| ---: | ---: | ---: | --- | --- | ---: | --- | ---: |
| 256 | 1 | 0.06 | 169.4, 170.3 | 169.9, 166.1 | -1.1% | 5.84 → 5.91 | +1.3% |
| 256 | 4 | 0.23 | 619.1, 624.3 | 643.4, 633.1 | +2.7% | 6.32 → 6.17 | -2.4% |
| 256 | 16 | 0.94 | 1,741.4, 1,753.5 | 2,025.9, 1,986.2 | +14.8% | 8.80 → 7.60 | -13.6% |
| 256 | 32 | 1.88 | 2,443.3, 2,465.9 | 3,096.8, 3,024.2 | +24.7% | 12.12 → 9.68 | -20.1% |
| 1,024 | 1 | 0.15 | 160.3, 161.2 | 164.9, 161.3 | +1.5% | 6.15 → 6.06 | -1.5% |
| 1,024 | 4 | 0.59 | 485.1, 486.8 | 545.1, 534.8 | +11.1% | 7.85 → 6.99 | -10.9% |
| 1,024 | 16 | 2.35 | 976.5, 987.8 | 1,287.9, 1,257.2 | +29.6% | 14.46 → 10.79 | -25.4% |
| 1,024 | 32 | 4.70 | 1,177.3, 1,188.9 | 1,680.8, 1,634.8 | +40.1% | 23.39 → 15.83 | -32.3% |
| 2,048 | 1 | 0.26 | 146.1, 147.0 | 155.1, 152.3 | +4.9% | 6.62 → 6.30 | -4.9% |
| 2,048 | 4 | 1.06 | 368.6, 372.5 | 450.7, 437.7 | +19.9% | 9.81 → 8.13 | -17.1% |
| 2,048 | 16 | 4.23 | 613.1, 621.8 | 867.1, 845.2 | +38.7% | 21.96 → 14.96 | -31.9% |
| 2,048 | 32 | 8.46 | 688.7, 697.2 | 1,029.5, 1,000.3 | +46.5% | 38.30 → 23.94 | -37.5% |
| 4,096 | 1 | 0.50 | 123.1, 124.2 | 137.2, 135.0 | +10.0% | 7.58 → 6.86 | -9.5% |
| 4,096 | 4 | 2.00 | 252.9, 254.9 | 326.9, 320.3 | +27.5% | 13.57 → 10.24 | -24.5% |
| 4,096 | 16 | 7.99 | 351.3, 352.5 | 510.8, 501.8 | +43.9% | 36.79 → 23.09 | -37.2% |
| 4,096 | 32 | 15.97 | 374.0, 376.5 | 564.6, 554.8 | +49.2% | 67.91 → 40.15 | -40.9% |

**The gain tracks KV bytes read per step.** Fitting median decode-step time against the KV bytes in flight at mid-decode (batch × (context + 128) tokens × bytes per token) over all 16 cells and both repeats:

| Variant | Step time model | R² | Effective KV read rate |
| --- | --- | ---: | ---: |
| BF16 | 5.84 ms + 4.02 ms per GB of KV | 1.000 | about 249 GB/s (L4 nominal about 300 GB/s) |
| FP8 (sat) | 5.97 ms + 4.44 ms per GB of FP8 KV | 0.999 | about 225 GB/s |

- **Fixed cost:** FP8 adds about 0.13 ms per step, from the clamp and cast kernels on every layer's K/V write.
- **Per-byte cost:** FP8 reads half the bytes, but each byte costs about 10% more time, from dequantizing inside the attention kernel.
- **Break-even:** FP8 is faster when 4.02·G − 4.44·(G/2) > 0.13, that is, when a step reads more than about **G ≈ 0.071 GB of BF16 KV**. That is about **616 tokens in flight across the batch** (at 114,688 bytes per token).
- **The one losing cell** is batch 1 × context 256 (about 384 tokens in flight, below break-even): −1.1% throughput and +1.3% step time. The FP8 repeat spread there (169.9 vs 166.1 tok/s) is larger than the difference, so it is a tie or a small loss, as predicted.
- **Upper bound:** as KV dominates, the step-time saving approaches 1 − 4.44 / (2 × 4.02) ≈ **45%**, or about 1.8× decode throughput. The largest cell (batch 32 × context 4,096, about 16 GB of BF16 KV) reached −40.9% step time and +49.2% throughput.

**Practical rule for this model, GPU, and engine** (in BF16 KV per decode step; 1 GB ≈ 8,700 tokens in flight across the batch):
- Below about 0.07 GB (about 616 tokens), FP8 is neutral or slightly slower.
- Up to about 0.3 GB (about 2,600 tokens), the gain is under 5%.
- At about 0.5 GB (about 4,400 tokens), expect about +10%.
- At 2–8 GB (17,000–70,000 tokens), expect +25% to +46%.
- At about 16 GB (about 140,000 tokens), expect about +50%, near the 1.8× ceiling. Single short requests gain nothing and can lose slightly. The extra capacity, a separate benefit, matters only when the BF16 pool limits admission ([experiment 5](../5_fp8_kv_capacity_w2/README.md): +10.5% on W2).

**Limits.**
- One model (Qwen3-0.6B, GQA 16/8, CUDA-core decode), one GPU (L4), random-token prompts, greedy decoding, 256 output tokens, offline.
- Step times come from decode-schedule timestamps under the overlap scheduler (steady-state rate), not per-kernel traces.
- A larger model with more weight bytes per step would push the break-even point higher, and a tensor-core decode path could change the per-byte cost. Neither was tested.

**Cost:** 4 containers; the exp 6 + exp 7 batch cost $0.119539 actual.
