# Experiment 4 findings: SGLang has the same latent FP8 overflow, masked by its decode-kernel choice (2026-10-24)

**Update 2026-10-25 ([experiment 8](../8_sglang_fp8_upstream_check/FINDINGS.md)):** the NaN-on-overflow cast is PyTorch behavior through **2.12.0**. From **2.13.0**, eager BF16 → e4m3fn casts saturate to ±448 on CPU and CUDA. Everything below holds for SGLang 0.5.9, which ships torch 2.9.1. SGLang main pins torch 2.14.1, so its unclamped MHA writer is safe at runtime, and the "latent bug" applies to releases that ship torch ≤ 2.12. The upstream comment draft was rewritten accordingly.

**Result.** SGLang 0.5.9 with `--kv-cache-dtype fp8_e4m3` and default scale 1.0 writes the **same NaN bytes** into its KV cache as mini-SGLang v1: layer-0 K reaches 516, and 8 values exceed about 464. Its default output is still correct, because SGLang always decodes FP8 KV with FlashInfer's **tensor-core** kernel, which tolerates those bytes. When the documented `SGLANG_FLASHINFER_USE_TENSOR_CORE=false` forces the **CUDA-core** decode kernel (the one mini-SGLang uses for Qwen3-0.6B), SGLang fails **byte-for-byte like mini-SGLang v1**: output hash `649e407e`, 999/1,000 tokens equal to `0`, and the same two fixtures collapsing with the same counts (70 and 89).

Raw records are in `results/long-context-probe/sglang-fp8-kv-reproduction/`. These are offline, single-container engine runs on one Modal L4 with radix cache disabled. Timings are unpaired and are not a performance result.

## 1. Reproduction run (`20261024T051039Z-base-auto-fp8_e4m3-fp8_e5m2.json`)

The image is `lmsysorg/sglang:v0.5.9-cu129-amd64`, unmodified, running torch 2.9.1+cu129 with the `flashinfer` attention backend.

| KV dtype | 1,024 → 1,000 hash | 8,000 → 1,000 hash | Token-0 runs (random, fixtures) | Pool tokens | First divergence vs SGLang BF16 (8 fixtures) |
| --- | --- | --- | --- | ---: | --- |
| `auto` (BF16) | `0941996b` | `cb86f41c` | none, none | 152,308 | — |
| `fp8_e4m3` | `0941996b` | `3fb5d39b` | none, none | 304,616 | `[118, 34, 94, 6, 24, none, 47, 15]` |
| `fp8_e5m2` | `0941996b` | `19aaa72c` | none, none | 304,616 | `[1, 26, 46, 2, 3, none, 11, 3]` |

- **Cross-engine matches.** SGLang BF16 reproduces mini-SGLang BF16 exactly at both lengths. SGLang `fp8_e4m3` at 8,000 reproduces mini-SGLang **v2 (saturating)** exactly (`3fb5d39b`), and its 1,024 output equals BF16, as v2's does. mini-SGLang v1 collapsed to `649e407e` at both lengths.
- **SGLang logged** "Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0. This may lead to less accurate results!"
- **e5m2 (option D in experiment 3):** no overflow, but it drifts from BF16 much earlier (tokens 1–3 on four fixtures) than e4m3. Its 2 mantissa bits cost visible precision. This rules e5m2 out as the preferred fix.
- **Capacity:** FP8 doubles SGLang's pool, from 152,308 to 304,616 tokens at `mem_fraction_static=0.8`.

## 2. Is PyTorch's CUDA cast saturating? No (`cast_probe.json`)

In torch 2.9.1+cu129 on an L4, a BF16 → `float8_e4m3fn` cast gives 448 for 440/450/460 and **NaN** for 470, 516, and −516. This holds for contiguous, mini-SGLang-style strided-slice, and transposed inputs, and for float32 input; CPU behaves the same. Clamp-then-cast gives ±448. Memory layout does not explain the difference.

## 3. Is the image's code different from the wheel? No

`/sgl-workspace/sglang` in the image is commit `bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2` (tag `v0.5.9`). Its `MHATokenToKVPool.set_kv_buffer` divides by `k_scale` only when one is given, then calls plain `.to(self.dtype)`, matching the wheel. Qwen3's fused RoPE + KV write (`enable_fused_set_kv_buffer`) is BF16-only, so the FP8 run went through this cast.

## 4. Layer-0 K range in SGLang (`20261024T052046Z-probe-fp8_e4m3.json`)

The probe-patched image logs layer-0 max |K| right before the cast, on eager calls only (CUDA-graph replays skip Python). On the 1,024-token prompt it logged `k_max=516.0 k_over448=23 k_over464=8`, exactly mini-SGLang's diagnostic: 23 values above 448, of which 8 become NaN under the non-saturating cast. The 8,000-token prompt's first 2,048-token prefill chunk had 53 values above 448 and 20 above 464. The output stayed correct, matching section 1. So SGLang **does** store NaN bytes, and the difference is in how they are read.

## 5. Kernel choice is the mask (`20261024T053237Z-cudacore-fp8_e4m3_cudacore.json`)

- **Source** (`layers/attention/flashinfer_backend.py::should_use_tensor_core`, 0.5.9): for FP8 KV it returns `True` unconditionally. For BF16 it requires a GQA group of at least 4; Qwen3-0.6B has 16/8 = 2. mini-SGLang (`attention/fi.py::use_tensor_cores`) applies only the GQA ≥ 4 rule, so it decoded FP8 with the CUDA-core `BatchDecodeWithPagedKVCacheKernel`.
- **Test:** the same base image and prompts, `fp8_e4m3`, with `SGLANG_FLASHINFER_USE_TENSOR_CORE=false`.

| | 1,024 → 1,000 | 8,000 → 1,000 | Fixture token-0 runs |
| --- | --- | --- | --- |
| SGLang fp8_e4m3, default (tensor-core decode) | `0941996b` (= BF16), 0 zeros | `3fb5d39b`, 0 zeros | all 0 |
| SGLang fp8_e4m3, CUDA-core decode forced | **`649e407e`, 999 zeros** | **`649e407e`, 999 zeros** | **[0, 70, 0, 0, 89, 0, 0, 0]** |
| mini-SGLang v1 (experiment 3) | `649e407e` | `649e407e` | 70 and 89 on the same prompts |

Prefill is also consistent: mini-SGLang v1's first token was always correct, even though its paged prefill (a tensor-core FA2 kernel) read the NaN-containing cache. The failure starts at the first CUDA-core decode step. **Hypothesis, not verified:** FlashInfer's tensor-core path dequantizes FP8 with a bit-manipulation conversion that maps the e4m3fn NaN encoding (`0x7F`/`0xFF`) to a finite value, while the CUDA-core decode path converts it to a real NaN. Confirming this needs a FlashInfer source reading or a kernel-level test.

## 6. The saturating clamp fixes SGLang (`20261024T182626Z-sat-fp8_e4m3_cudacore-fp8_e4m3.json`, 2026-10-24)

The `sat` image clamps K/V to ±448 before the cast in the installed 0.5.9 `MHATokenToKVPool.set_kv_buffer` (study patch, mirroring PR #41732's MLA change). One container ran:

| SGLang 0.5.9 + clamp | 1,024 → 1,000 | 8,000 → 1,000 | Fixture token-0 runs |
| --- | --- | --- | --- |
| fp8_e4m3, CUDA-core decode forced | `0941996b` (= BF16), 0 zeros | `3fb5d39b`, 0 zeros | all 0 |
| fp8_e4m3, default tensor-core decode | `0941996b`, 0 zeros | `3fb5d39b`, 0 zeros | all 0 |

- The configuration that collapsed in section 5 now matches BF16 at 1,024 tokens and mini-SGLang v2 at 8,000. **The clamp removes the failure in SGLang itself.**
- With default decode, the clamp leaves 6 of 8 fixtures token-identical to unpatched SGLang FP8. It changes only the sky and TCP prompts, the two that collapsed in v1, so they contain the outlier keys. This supports, but does not prove, the section-5 hypothesis: the tensor-core path reads NaN bytes as some finite value other than 448, so clamping to exactly 448 shifts those two continuations.
- CUDA-core + clamp differs from tensor-core + clamp on 6 of 8 fixtures. That is expected from the different kernels' accumulation order; all are clean, with no token-0 runs.

## What this means

- **SGLang 0.5.9 (torch 2.9.1) has a latent correctness bug.** `MHATokenToKVPool.set_kv_buffer` writes NaN for any |K| or |V| above about 464 (above about 464·s with a scale). The default tensor-core decode hides it. Any path that decodes FP8 KV with a kernel that honors NaN bytes, starting with the documented `SGLANG_FLASHINFER_USE_TENSOR_CORE=false`, produces garbage on Qwen3-0.6B. Whether other attention backends (triton, fa3, trtllm on other GPUs) honor the NaN bytes is untested.
- **The fix is known and small:** saturate before the cast. Main already does this in `set_kv_buffer_prefix_valid`, and open PR [#41732](https://github.com/sgl-project/sglang/pull/41732) does it for MLA writers, but neither covers `MHATokenToKVPool.set_kv_buffer`. mini-SGLang v2 shows the clamp removes the failure, and SGLang's default FP8 output equals that clamped output exactly.
- The e5m2 format avoids overflow but drifts from BF16 within 1–3 tokens on half of the fixtures, so it is not a good substitute.

## Next steps

1. ~~Run the clamp fix in SGLang with CUDA-core decode forced.~~ **Done 2026-10-24; it passed** (section 6). The earlier hold was a billing-report artifact: CSV output wraps at 80 columns, and the JSON report showed $1.07 actual spend.
2. **SGLang main (source only, `773cb938`):** the source is unchanged, but main pins torch 2.14.1, whose cast saturates ([experiment 8](../8_sglang_fp8_upstream_check/FINDINGS.md)), so it is not affected at runtime. `MHATokenToKVPool.set_kv_buffer` still casts without saturation, and `should_use_tensor_core` still forces tensor-core decode for FP8 with the same `SGLANG_FLASHINFER_USE_TENSOR_CORE` override. Main has not been run; a runtime check needs a newer image (an optional, paid step).
3. **Upstream (the user's decision):** post the evidence (sections 4–6 and the experiment-3 diagnostic) as a comment on #41732, asking whether to extend the fix to `MHATokenToKVPool.set_kv_buffer` there or in a follow-up PR. Keep the claim narrow: Qwen3-0.6B, L4, SGLang 0.5.9, scale 1.0, CUDA-core decode. A comment is drafted in UPSTREAM_COMMENT_DRAFT.md (private draft). Nothing has been posted.

## Cost

This experiment ran four GPU containers (base, cast probe, K probe, CUDA-core run) plus a no-GPU image-source check and two CPU image builds. The worst-case reservations are on the status board. The billing report at about 05:40 UTC was malformed and showed most new apps at $0.0; only `fp8-cast-probe` had posted, at $0.0048. All apps were stopped. Reconciled with `--json` on 2026-10-24: this app's total is $0.228946, including the section-6 run ($0.059).
