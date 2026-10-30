# FP8 KV-cache feasibility: findings (2026-10-23, v2 2026-10-24)

**v2 update:** a saturating cast (clamp to ±448) removes v1's NaN failure. Outputs are coherent, 1,024-input output is identical to BF16, decode attention is −42.3% per step, and whole-call time is −13.7% (2 vs 2 runs). See [v2 results](#v2-saturating-cast-results-2026-10-24). The v1 sections below are kept as recorded.

**Result: the unscaled FP8 (e4m3, scale 1.0) KV path is mechanically faster but numerically broken for this workload.** It halves KV bytes per token, doubles pool token capacity, and cuts decode-attention kernel time by 43%. However, some layer-0 key values exceed FP8's ±448 range, and casting them produces NaN. The NaN then spreads, and every token generated after the first is `0` (`!`). The timing below is therefore **not a valid speedup claim**; it measures kernels running on partly-NaN data. A correct FP8 treatment needs saturation or scaling; see "Solutions considered" and "Plan".

All runs are offline `LLM.generate` timings on one Modal L4. Source: `sgl-project/mini-sglang@9a91cfaf`, plus [`fp8_kv_cache.patch`](fp8_kv_cache.patch) (sha256 `34379d45…`) for FP8. Both images pin torch 2.9.1, flashinfer-python 0.7.2.post1, sgl-kernel 0.3.21, and transformers 4.57.3; each `run.json` records the installed versions. Raw records are in `results/long-context-probe/fp8-kv-cache-feasibility/`; reproduce the table with `uv run python projects/long-context-probe/3_fp8_kv_cache_feasibility/table.py`.

## Runs

| Order | Variant | Input / output | Elapsed (s) | KV dtype | Pool tokens | Output hash | Run ID |
| ---: | --- | --- | ---: | --- | ---: | --- | --- |
| pilot | FP8 | 1,024 / 1,000 | 8.594 | float8_e4m3fn | 345,344 | `649e407e` | `20261023T202859Z-143dc4c3-pilot-fp8-1024-r1` |
| 1 | BF16 | 8,000 / 1,000 | 10.600 | bfloat16 | 172,544 | `cb86f41c` | `20261023T203029Z-fe7b5624-final-bf16-8000-r1` (trace) |
| 2 | FP8 | 8,000 / 1,000 | 9.354 | float8_e4m3fn | 345,344 | `649e407e` | `20261023T203152Z-aeff0f1e-final-fp8-8000-r1` (trace) |
| 3 | FP8 | 8,000 / 1,000 | 8.474 | float8_e4m3fn | 345,344 | `649e407e` | `20261023T203736Z-2b7166ae-final-fp8-8000-r2` |
| 4 | BF16 | 8,000 / 1,000 | 9.860 | bfloat16 | 172,544 | `cb86f41c` | `20261023T204459Z-2dc79333-final-bf16-8000-r2` |

Two earlier pilot attempts failed in the runner, before any measurement. The first hashed the patch at import inside the container; the second read the KV pool from the wrong attribute. Their `run.json` records are kept with `status: failed`. Every listed run produced 1,000/1,000 tokens.

**Reproducibility check.** The fresh BF16 control's output hash `cb86f41c…` equals experiment 2's 8,000-input hash, despite the newly pinned dependency set.

**Red flag.** FP8 produced the **same output hash for the 1,024- and 8,000-token prompts**; BF16 gives a different hash at every length. That means the FP8 output did not depend on the long prompt.

## Memory

| | BF16 | FP8 |
| --- | ---: | ---: |
| KV bytes per token (28 layers, K+V) | 114,688 | 57,344 |
| Logical KV for 8,000 + 1,000 tokens | 0.961 GiB | 0.481 GiB |
| Allocated pool | 18.457 GiB | 18.457 GiB |
| Pool capacity (tokens) | 172,544 | 345,344 (2.0×) |

The engine sizes the pool from free memory, so FP8 keeps the same bytes and doubles token capacity. One request uses neither, so capacity gains need a concurrency experiment.

## Which step changed: matched 128-output traces

`uv run python projects/long-context-probe/3_fp8_kv_cache_feasibility/compare_traces.py <bf16 trace> <fp8 trace>`; the output is saved as `results/.../fp8-kv-cache-feasibility/trace_comparison.json`. Both traces have 127 inferred decode steps.

| Per decode step, first-32 median | BF16 | FP8 | Change |
| --- | ---: | ---: | ---: |
| Decode attention (`BatchDecodeWithPagedKVCacheKernel`, 28 layers) | 3.710 ms | 2.121 ms | **−42.8%** |
| Decode-step start interval | 9.209 ms | 7.774 ms | −15.6% |
| Main cuBLAS GEMV | 3.750 ms | 3.775 ms | +0.7% |

| Whole-trace kernel totals | BF16 | FP8 | Δ |
| --- | ---: | ---: | ---: |
| Decode attention | 474.4 ms | 270.9 ms | −203.6 ms |
| New BF16→FP8 casts (`vectorized_elementwise_kernel`) | 28 launches, 0.03 ms | 7,140 launches, 9.98 ms | +9.9 ms |
| Prefill attention (`BatchPrefillWithPagedKVCacheKernel`) | 4.15 ms | 2.35 ms | −1.8 ms |

Decode attention is the step that improves, consistent with it being bound by KV bytes read. The added cost is two small cast kernels per layer per forward pass, about 0.08 ms per step. The matrix work does not change. Caveat: FlashInfer's kernel time should not depend on whether values are NaN, but this trace ran on partly-NaN KV. Kernel sums can overlap, and the trace covers one profiled run per variant.

## Whole-call time (not valid as a speedup; see correctness)

The BF16 mean is 10.230 s (two runs, 0.74 s apart). The FP8 mean is 8.914 s (two runs, 0.88 s apart), **12.9% lower**. The spread within each variant is more than half the difference between them, so this is weak evidence even before the correctness failure.

## Correctness: why FP8 fails

A diagnostic stage (`--stage diagnose`) zeroed the pool, generated 32 tokens from the 1,024-token prompt, and scanned the stored KV. Results are in `results/.../fp8-kv-cache-feasibility/diagnose/`.

- BF16 layer-0 **K reaches 516**, with 23 values above 448. Other layers' K stays below 224, and V stays below 90.
- FP8's `float8_e4m3fn` maximum is 448, and it has no infinity. PyTorch's `.to(float8_e4m3fn)` does not saturate: values just above 448 round to 448, and larger values become **NaN**. In layer 0, 8 K bytes became NaN.
- Attention over a NaN key produces NaN. In layers 1–27, exactly **31,744 NaNs** each appear in K and V. That is 31 generated tokens × 8 heads × 128 dims, so every token after the first is entirely NaN.
- The first output token matches BF16 (`198`). Every later token is `0` (`!`), which is what greedy argmax returns from NaN logits. This explains the identical FP8 hashes.

**Natural-prompt quality** (8 prompts, greedy, 128 tokens, compared with the BF16 run 1): first divergent token indexes were `[5, 15, none, 6, 28, none, 47, 20]`. Six prompts diverge and stay fluent ("Paris. The capital of Italy is Rome…" versus BF16's "Paris. The capital of France is also…"). **Two prompts (sky, TCP) collapse into 70 and 89 trailing `!` tokens**, the NaN signature, so the overflow is not limited to random-token prompts. The BF16 baseline itself repeats on several raw prompts, since no chat template is used. These fixtures detect breakage; they are not an accuracy benchmark.

## Prediction check

| Prediction (user, before runs) | Observed | Verdict |
| --- | --- | --- |
| 1. Decode attention 40–50% lower | −42.8% per step | Mechanically met, on NaN-contaminated KV |
| 2. Whole call ≥15% faster | −12.9%; spread comparable to the difference | Not met, and invalid due to NaN output |
| 3. Diverges late, stays coherent | Long prompts: NaN collapse after token 1; natural: 6/8 coherent divergence at tokens 5–47, 2/8 collapse | Falsified for this unscaled path |
| 4. Logical KV halves, pool bytes equal, capacity ~2× | 0.961 → 0.481 GiB; 18.457 GiB both; 172,544 → 345,344 | Met |

## v2 (saturating cast) results, 2026-10-24

The clamp-only patch [`fp8_kv_cache_v2_saturate.patch`](fp8_kv_cache_v2_saturate.patch) (sha256 `1140df00…`) **removes the failure and keeps the speed mechanism.** The BF16 control reused the unmodified-upstream image, with the same pins as v1.

| Order | Variant | Input / output | Elapsed (s) | Output hash | Run ID |
| ---: | --- | --- | ---: | --- | --- |
| pilot | fp8_sat | 1,024 / 1,000 | 8.564 | `0941996b` (= BF16) | `20261024T044417Z-870bf4dc-pilot-fp8_sat-1024-r1` |
| 1 | BF16 | 8,000 / 1,000 | 9.842 | `cb86f41c` | `20261024T044839Z-f48385d3-final-bf16-8000-r1` (trace) |
| 2 | fp8_sat | 8,000 / 1,000 | 8.772 | `3fb5d39b` | `20261024T044925Z-3a52b0d2-final-fp8_sat-8000-r1` (trace) |
| 3 | fp8_sat | 8,000 / 1,000 | 8.198 | `3fb5d39b` | `20261024T045014Z-d42580e5-final-fp8_sat-8000-r2` |
| 4 | BF16 | 8,000 / 1,000 | 9.818 | `cb86f41c` | `20261024T045100Z-b70c49e8-final-bf16-8000-r2` |

**Correctness.**
- The diagnostics (`diagnose/20261024T044749Z-bf16.json`, `…044834Z-fp8_sat.json`) show 23 BF16 layer-0 K values above 448 (max 516). These are the clipped values. FP8 stores **0 NaN bytes**, with layer-0 K capped at 448.0, and the first 32 tokens match BF16.
- At 1,024 input, all 1,000 `fp8_sat` tokens are identical to BF16: the hash equals experiment 2's BF16 1,024 hash.
- At 8,000 input, the output is prompt-dependent but not identical to BF16. Only hashes are stored, so the divergence index for the 8,000-token output is not measured.
- All 8 natural prompts are coherent, with zero token-`0` runs. The first divergences from BF16 are `[5, 15, none, 6, 28, none, 47, 20]`, the same indexes as v1. The drift therefore comes from FP8 rounding (for example, 89.5 → 88.0), not from clipping. The sky and TCP prompts that collapsed in v1 now finish normally.
- Logits were not instrumented; zero NaN in the KV plus no token-`0` runs stands in for finite logits. Eight prompts detect breakage; they are not an accuracy benchmark.

**Which step changed** (`trace_comparison_v2.json`, first 32 decode steps):

| Per decode step | BF16 | fp8_sat | Change |
| --- | ---: | ---: | ---: |
| Decode attention (28 layers) | 3.734 ms | 2.153 ms | **−42.3%** |
| Decode-step start interval | 9.361 ms | 8.000 ms | −14.5% |
| Main GEMV | 3.793 ms | 3.807 ms | +0.4% |

Across the trace, decode attention saves 202.3 ms. The clamp and cast add 14,308 elementwise launches (23.1 ms, about 0.18 ms/step), twice v1's cast-only overhead. That is a candidate for fusing into the store kernel.

**Whole call.** The BF16 mean is 9.830 s (two runs 0.024 s apart); the `fp8_sat` mean is 8.485 s (0.574 s apart): **−13.7%**. The repeat spread is smaller than the difference, but there are only two repeats per variant, and the BF16 run 1 here (9.84 s) was 0.76 s faster than v1's BF16 run 1 on 2026-10-23. Treat this as a measured, single-request offline result, not a general speedup.

## Accuracy eval (2026-10-24)

`--stage eval` asks 200 held-out two-digit arithmetic questions (seed 20261024: 69 `+`, 63 `−`, 68 `×`), each "What is a op b? Answer with only the number.", through Qwen3's chat template with thinking disabled. Decoding is greedy with at most 24 tokens. Each variant runs in its own container on its experiment-3 image. The authoritative scorer is [`score_eval.py`](score_eval.py), which takes the **last** integer in the answer. The `accuracy` stored in the 2026-10-24 records came from a first-integer parse that scored answers like "85 * 32 = 2720" as 85; ignore it.

| KV cache | Correct | × | − | + | Outputs with token-0 runs | Run ID |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 | **182/200 (91.0%)** | 53/68 | 61/63 | 68/69 | 0 | `20261024T183821Z-eval-bf16` |
| FP8 v2 (saturating) | **180/200 (90.0%)** | 50/68 | 61/63 | 69/69 | 0 | `20261024T183917Z-eval-fp8_sat` |
| FP8 v1 (plain cast) | 1/200 (0.5%) | 0/68 | 1/63 | 0/69 | 200 | `20261024T183949Z-eval-fp8` |

- **v2 vs BF16:** the text is identical on 174/200 problems. Both are right on 178; BF16 alone on 4 and FP8 alone on 2. An exact McNemar test on those 6 disagreements gives p ≈ 0.69, so this eval detects **no accuracy difference**. Most differing outputs differ only in format (`×` vs `*`) or in one multiplication digit.
- **v1** corrupts every output ("8!!!!…"), matching the experiment-3 diagnosis. This is the same failure seen through a real task.
- **Limits:** 200 short arithmetic prompts with short contexts and one model. This is a sanity check for task accuracy, not a broad benchmark, and it does not test long-context retrieval or calibration.

## Is FP8 KV a failure? What this run does and does not show

The v1 patch **is not usable**: it would fail on real traffic, since 2 of 8 ordinary prompts collapsed. FP8 KV as a technique is not ruled out, though. Production engines (vLLM, full SGLang, TensorRT-LLM) ship it with two safeguards v1 lacked.

1. **A scale factor.** FP8 stores `x / s` and reads back `fp8 × s`, so its usable range is ±448·s, not ±448. FlashInfer's `run(..., k_scale=, v_scale=)` already applies the scale: it multiplies `k_scale` into the softmax scale and `v_scale` into the output. v1 passed nothing, so the scale was 1.0.
2. **A saturating cast.** Out-of-range values are clipped to ±448 instead of becoming NaN. PyTorch's `.to(torch.float8_e4m3fn)` does not saturate.

## Solutions considered

| Option | How values above 448 are handled | Diff size | Quality risk |
| --- | --- | --- | --- |
| **A. Saturating cast (clamp)** | Clamp K/V to ±448 before the cast; never NaN | 1–2 lines in `mha_pool.store_kv` | Clips only rare outliers (23 of about 1.08 million layer-0 K values written in the diagnostic: 1,056 tokens × 8 heads × 128 dims) |
| **B. Static per-layer scales** | Calibrate each layer's max \|K\|, \|V\| on BF16 prompts with a margin; store `K / s`; pass `s` to FlashInfer | About 10 lines plus a calibration step | Low, unless real inputs exceed the calibrated range |
| **C. B + A** | Scale for the normal range, clamp as a safety net | About 11 lines | Lowest of the simple options; the production pattern |
| D. e5m2 format | Range ±57,344, but 2 mantissa bits instead of 3 | 1 line | Solves range, much coarser rounding |
| E. Layer 0 in BF16 | Only the overflowing layer stays BF16 | Large: the pool would need mixed dtypes | Good, but complex |
| F. Per-token/block scales | Fine-grained scales stored beside the KV | Large; FlashInfer's FA2 paged path does not support it | Best, out of scope |

One scale for the whole model would be poor. Layer-0 K reaches 516, while every other layer's K stays at or below about 221, and V peaks at 89.5. A scale large enough for layer 0 would waste precision in every other layer. Per-layer scales fit naturally because attention is called once per layer.

## Plan (decided 2026-10-24)

1. **v2 = option A, clamp only.** This tests whether range was the *only* problem. **Done 2026-10-24; it passed** (see [v2 results](#v2-saturating-cast-results-2026-10-24)).
2. Protocol, same as v1: pilot, BF16/`fp8_sat` diagnostics, then BF16 → fp8_sat → fp8_sat → BF16 with traces. **Done.**
3. Success criteria for v2: 0 NaN bytes, no token-`0` runs, prompt-dependent hashes, coherent text. **All met.**
4. v2 is a diagnostic for the range question, not the final design. Logits are not instrumented.

## How full SGLang handles this (source reading, 2026-10-24; not yet run)

Read from the `sglang==0.5.9` wheel (the serving track's pin) and from `sgl-project/sglang` main at `7d996300b98e3f74a7e79adefbac4d95de0da508` (2026-10-24). These are code observations only; nothing was executed.

- **0.5.9, `mem_cache/memory_pool.py::MHATokenToKVPool.set_kv_buffer`:** if K/V dtype differs from the cache, it divides by `k_scale`/`v_scale` when they are not `None`, then calls plain `cache_k.to(self.dtype)`. There is **no clamp**, the same pattern as our v1. The store kernel after it is a byte copy.
- **0.5.9, scale defaults:** `RadixAttention` initializes `k_scale = None`. With `--kv-cache-dtype fp8_e4m3` and no `--quantization-param-path`, `model_runner.py` only logs "Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0. This may lead to less accurate results!" Qwen3 implements `load_kv_cache_scales`, so a per-layer scale JSON is supported. Even with scales there is no clamp, so any value above 448·s would still become NaN.
- **main:** the same plain cast remains in `MHATokenToKVPool.set_kv_buffer` (and `MHATokenToKVPoolFP4`). A newer `set_kv_buffer_prefix_valid` path and its fused Triton kernel **do saturate** (`saturate_to_fp8_range`, `FP8_MAX`), with the comment "Saturate like the fused kernel above". SGLang has recognized the hazard but does not apply it on every write path.
- **Implication (hypothesis):** SGLang 0.5.9, and possibly main, with `--kv-cache-dtype fp8_e4m3` and default scales could hit our v1 failure on Qwen3-0.6B when layer-0 K exceeds about 464. Its warning says "less accurate", but our evidence shows NaN collapse. Model code, the attention backend, or the prefill path could differ, so this must be reproduced before any upstream claim. **Upstream search (2026-10-24):** open PR [#41732](https://github.com/sgl-project/sglang/pull/41732) ("Saturate bf16->e4m3 KV/QKV casts to ±448 instead of overflowing to NaN", opened 2026-09-29, unmerged, GPU tests not yet run) fixes this class of bug for **MLA** writers only. `MHATokenToKVPool.set_kv_buffer`, used by Qwen3, is unchanged in its diff. No duplicate MHA report was found; that is a search result, not proof.

## Next steps: outcomes (updated 2026-10-24)

1. **Full-SGLang reproduction: done, as [experiment 4](../4_sglang_fp8_kv_reproduction/FINDINGS.md).** SGLang 0.5.9 writes the same NaN bytes, and its default FP8 output is correct only because it always uses tensor-core decode for FP8 KV. Forcing CUDA-core decode reproduces v1's failure byte-for-byte. Remaining: run the clamp in SGLang under CUDA-core decode, check SGLang main, then (the user's call) post evidence on PR #41732.
2. **FP8 capacity on the W2 benchmark: done, as [experiment 5](../5_fp8_kv_capacity_w2/README.md).** Throughput rose 62.6% (1,818 → 2,956 tok/s). The capacity-matched control attributes +47.6% to cheaper attention at identical batching and +10.5% to doubled concurrency.
3. **v3 calibrated per-layer scales: deferred, with weaker motivation.** The arithmetic eval found v2 within noise of BF16 (90.0% vs 91.0%). v3 would matter only if a broader or long-context evaluation shows a v2 accuracy gap.
4. **Fusing the clamp and cast into the store kernel: deferred.** It saves about 0.18 ms/step and needs a custom CUDA kernel, which the project instructions exclude without a recorded reason.

Keep v1 timings labeled invalid in every summary.

## Cost and cleanup

This experiment launched 9 containers in total: two failed pilot starts, one pilot, four final runs, and two diagnostics. The pre-launch reservations were $1.70 plus $0.68. The Modal billing report pulled at about 21:00 UTC showed $0.0 for every `long-context-fp8-kv-cache` app (charges not yet posted); recheck later. `modal app list` showed all apps stopped.
