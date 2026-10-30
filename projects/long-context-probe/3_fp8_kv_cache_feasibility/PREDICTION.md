# BF16 versus FP8 KV-cache prediction

Written 2026-10-23, **before** any FP8 execution (local or Modal). The user chose to run this experiment before finishing the mini-SGLang map and CPU toy engine, so it is an early exploratory tweak, not the W5 tweak selection. Keep this prediction unchanged; append results below.

## Known starting point

- BF16 control: unmodified `sgl-project/mini-sglang` at `9a91cfafe754aa85daee49998176275667eb58f2`.
- FP8 treatment: the same commit plus [`fp8_kv_cache.patch`](fp8_kv_cache.patch), currently applied as an uncommitted diff in `~/projects/upstream/mini-sglang-fp8-kv` (four files, 25 added lines). KV is stored as `float8_e4m3fn` with scale 1.0 (a direct cast, no calibration). Weights, activations, and queries stay BF16. FlashInfer dequantizes KV inside its attention kernels. The runner explicitly requests `kv_cache_dtype="fp8_e4m3"`; selecting the fork alone does not activate FP8.
- On an L4 (sm89), mini-SGLang auto-selects the FlashInfer (`fi`) FA2 backend for both prefill and decode. The patch rejects FP8 with any other backend.
- For Qwen3-0.6B (28 layers, 8 KV heads, head dimension 128), 9,000 logical KV tokens hold `2 × 28 × 8 × 128 × 9,000` elements: **1,032,192,000 bytes (0.96 GiB) in BF16** and **516,096,000 bytes (0.48 GiB) in FP8**.
- At 8,000 context, one decode step's attention reads about 917 MB of BF16 KV. The L4's memory bandwidth is roughly 300 GB/s. In the experiment-2 trace, decode attention took about 3.7 ms/step at 8,000 tokens and the main GEMV about 3.8 ms/step. Whole-call time was 9.81 s.
- The engine sizes its KV pool from free memory. FP8 should keep the allocated pool bytes about the same and roughly double its token capacity. A single request does not use that extra capacity.

## Feasibility gate

- Source diff: `fp8_kv_cache.patch` changes only the engine dtype selection and page arithmetic, the pool write, and FlashInfer planning. Upstream `store_cache` is a raw byte copy that checks K/V against the cache dtype, so BF16 K/V must be cast before the store.
- Backend support: FlashInfer 0.7.2.post1 accepts separate `q_data_type` and `kv_data_type` for both the paged prefill and decode wrappers. All GPU execution runs on Modal L4s (user decision, 2026-10-23); the 1,024-token FP8 pilot is the compatibility smoke test.
- Quality method: greedy decoding (temperature 0) of fixed natural prompts. Measure the first token where FP8 diverges from BF16 and read the decoded text.

## Prediction (user, 2026-10-23)

1. **Decode-attention kernel time per step at ~8,000 context: about 40–50% lower** with FP8. Reason: decode attention is memory-bandwidth-bound, so halving KV bytes nearly halves its time.
2. **Whole-call time at 8,000 input / 1,000 output: at least 15% faster.**
3. **Quality: outputs diverge late and stay coherent.** Greedy outputs match BF16 for many tokens, then differ, and the text remains sensible.
4. Memory (from arithmetic, not a user guess): logical request KV halves, allocated pool bytes stay similar, and pool token capacity roughly doubles.

Falsifiers:

- Prediction 1 fails if the FP8 trace's decode-attention kernel sum per step is reduced by less than 40% (or by more than 50%) relative to the matched BF16 trace.
- Prediction 2 fails if the mean of the two FP8 runs is less than 15% below the mean of the two BF16 runs, or if the repeat spread is as large as the difference.
- Prediction 3 fails if most prompts diverge within the first 10 tokens or the text is visibly degraded. Identical outputs on every prompt would also not match "diverges late."
- Prediction 4 fails if the FP8 pool's token capacity does not rise by about 2× or if the dtype check shows BF16 storage.

Counters needed beyond whole-call time: matched 128-output traces with per-step decode-attention, prefill-attention, KV-store, and cast-kernel sums; pool bytes, capacity, and shape.

## Run plan and bounds

- Commands and order (FP8 pilot, then BF16–FP8–FP8–BF16): [runbook](README.md).
- Stop conditions: unsupported dtype or backend, short output, early or degraded divergence, a source diff wider than the patch, or the cost limit.

## Results (append after runs; do not rewrite the prediction)

Runs completed 2026-10-23 on Modal L4. Details, run IDs, and the diagnostic are in [FINDINGS.md](FINDINGS.md).

1. Decode attention per step: 3.710 → 2.121 ms (**−42.8%**), inside the predicted 40–50%, but measured on NaN-contaminated KV.
2. Whole call: BF16 mean 10.230 s, FP8 mean 8.914 s (−12.9%); repeat spreads of 0.74 and 0.88 s. **Not met**, and not a valid speedup because the FP8 output is broken.
3. Quality: **falsified** for unscaled FP8. Layer-0 K reaches 516, above e4m3's 448 limit; the non-saturating cast produces NaN; every generated token after the first collapses to `0` on long prompts and on 2 of 8 natural prompts.
4. Memory: **met**. Logical KV is 0.961 → 0.481 GiB, the pool is 18.457 GiB in both, and capacity rises 172,544 → 345,344 tokens.

What surprised: the speed mechanism behaved as predicted, while correctness failed on range rather than precision. A fix (saturating cast or per-tensor scales) is a new treatment and needs a new prediction.

---

# v2 (saturating cast) prediction

Written 2026-10-24, **before** any `fp8_sat` GPU run. Keep it unchanged; append results below.

## Treatment

[`fp8_kv_cache_v2_saturate.patch`](fp8_kv_cache_v2_saturate.patch) (sha256 `1140df00…`) is the full v1 diff plus a clamp of K and V to `torch.finfo(float8_e4m3fn).max` (448) before the cast in `MHAKVCache.store_kv`. The scale is still 1.0. The runner variant is `fp8_sat`; the BF16 control is the same unmodified-upstream image as v1. Rationale and alternatives: [FINDINGS.md](FINDINGS.md#solutions-considered).

## Prediction (user, 2026-10-24)

1. **Quality: all 8 natural prompts stay coherent, with no trailing `!` runs.** They drift from BF16 at points similar to v1's healthy prompts (tokens 5–47) or stay identical.
2. **Long random prompts: the output becomes prompt-dependent.** The 1,024 and 8,000 hashes differ from each other and from v1's constant `649e407e…`; they may still differ from BF16.
3. **Decode attention per step at 8,000 context: no change or slower** compared with BF16.
4. **Whole call at 8,000 input / 1,000 output: 10–15% faster** than BF16.

Falsifiers:

- Prediction 1 fails if any prompt has a run of token `0`, or if the text is visibly degraded.
- Prediction 2 fails if the `fp8_sat` pilot (1,024) and final (8,000) hashes are equal, or if either equals `649e407e…`.
- Prediction 3 fails if the `fp8_sat` decode-attention kernel sum per step is more than 10% below BF16 in the matched trace.
- Prediction 4 fails if the mean of the two `fp8_sat` runs is not 10–15% below the mean of the two BF16 runs, or if the repeat spread is as large as the difference.

The diagnostic must show **0 FP8 NaN bytes** for v2 to count as numerically valid at all.

## Run plan and bounds

`fp8_sat` pilot (1,024) → BF16 and `fp8_sat` diagnostics (1,024 / 32 output) → BF16, fp8_sat, fp8_sat, BF16 at 8,000 / 1,000 with traces on repeat 1. That is 7 containers, $2.38 worst case at $0.34 each. Stop if NaN bytes are nonzero, output is short, or the pilot text is degenerate.

## v2 results (append after runs; do not rewrite the prediction)

Runs completed 2026-10-24 on Modal L4; details are in [FINDINGS.md](FINDINGS.md#v2-saturating-cast-results-2026-10-24).

1. Quality: **met.** All 8 prompts are coherent with zero token-`0` runs in both final runs. The first divergences from BF16 are `[5, 15, none, 6, 28, none, 47, 20]`, identical to v1's indexes, so the drift comes from FP8 rounding, not from clipping.
2. Long prompts: **met.** The 1,024 hash `0941996b…` equals the BF16 1,024 hash, so all 1,000 tokens are identical to BF16. The 8,000 hash `3fb5d39b…` differs from BF16 (`cb86f41c…`), from the pilot, and from v1's constant `649e407e…`.
3. Decode attention: **falsified.** It is 3.734 → 2.153 ms per step (−42.3%), essentially v1's −42.8%. Kernel time depends on bytes read, not on the values.
4. Whole call: **met.** The BF16 mean is 9.830 s (runs 0.024 s apart); the `fp8_sat` mean is 8.485 s (0.574 s apart): **−13.7%**, with the repeat spread smaller than the 1.345 s difference.
5. Diagnostic: 0 FP8 NaN bytes; the clamp capped the 23 layer-0 K values above 448 (BF16 max 516).
