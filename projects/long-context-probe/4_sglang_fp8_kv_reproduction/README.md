# Experiment 4: does full SGLang reproduce the FP8 KV overflow?

**Question.** [Experiment 3](../3_fp8_kv_cache_feasibility/FINDINGS.md) found that unscaled e4m3 KV storage in mini-SGLang turns Qwen3-0.6B layer-0 K values above about 464 into NaN, and that a saturating cast fixes it. Source reading shows SGLang 0.5.9 and main also cast without saturation in `MHATokenToKVPool.set_kv_buffer`. Does the failure reproduce in the real engine?

**Upstream context (searched ).** Open PR [sgl-project/sglang#41732](https://github.com/sgl-project/sglang/pull/41732) ("Saturate bf16->e4m3 KV/QKV casts to ±448 instead of overflowing to NaN", opened, unmerged) adds `to_fp8_satfinite` but changes only **MLA** writers (`MLATokenToKVPool`, MLA quantize helpers, a JIT kernel, the TRT-LLM MLA backend). It does not touch the MHA writer used by Qwen3 and Llama-style models, and its GPU tests had not been run. A search did not find a duplicate MHA report; that does not prove none exists.

**Setup.** The official image is `lmsysorg/sglang:v0.5.9-cu129-amd64` (the serving track's pin), unmodified, with Qwen3-0.6B at revision `c1899de2…` from the shared volume and one L4. One container runs each KV dtype in its own subprocess: `auto` (BF16 control), `fp8_e4m3`, and `fp8_e5m2` (option D, wider range). Every engine uses `disable_radix_cache=True`, so the 1,024-token prompt cannot be reused by the 8,000-token prompt that extends it, plus `context_length=9216`, `mem_fraction_static=0.8`, and `cuda_graph_max_bs=8`. Prompts are the seeded random 1,024- and 8,000-token prompts from experiments 2–3 (1,000 greedy output tokens, `ignore_eos`) and the 8 natural fixtures (128 tokens). Timings are single, unpaired offline calls in one container; they are not a performance result.

The `sat` image applies a study patch to the installed 0.5.9 `MHATokenToKVPool.set_kv_buffer`: a clamp to ±448 before the cast, mirroring #41732's MLA change. It runs only if the base image reproduces the failure.

**Expected signal (agent-written; the user waived a formal prediction).** If SGLang behaves like v1, `fp8_e4m3` on the random prompts shows a long run of token `0` after the first token, an identical output hash for 1,024 and 8,000, and possibly `0` runs on the sky and TCP fixtures. `auto` and `fp8_e5m2` should not collapse. If `fp8_e4m3` stays clean, SGLang's path differs from mini-SGLang's (for example in where K is cast or what the backend does), and that becomes the finding.

## Why we ran this

Experiment 3's failure was in a 25-line study patch to a teaching engine. That alone says little about production. Full SGLang ships the same feature (`--kv-cache-dtype fp8_e4m3`, default scale 1.0), its source showed the same unsaturated cast, and an open upstream PR fixed only the MLA path. If the real engine had the bug, it would be a genuine upstream contribution. If it did not, the difference would explain what mini-SGLang got wrong. In both cases, the check had to run on the serving track's pinned image before anyone claimed anything upstream.

## What we did

1. **Base run:** one container on the unmodified `v0.5.9` image ran BF16 (`auto`), `fp8_e4m3`, and `fp8_e5m2` on the experiment 2–3 prompts. FP8 e4m3 output was correct, unlike mini-SGLang v1.
2. **Cast probe** ([`cast_probe.py`](cast_probe.py)): checked that torch 2.9.1's CUDA BF16 → e4m3 cast returns NaN above about 464 for contiguous, strided, and transposed inputs. Memory layout was not the reason.
3. **Probe image:** patched `set_kv_buffer` to log layer-0 max |K| before the cast. It logged 516, with 8 values above 464. So SGLang writes the same NaN bytes.
4. **CUDA-core decode:** read `should_use_tensor_core` and found that SGLang forces FlashInfer's tensor-core decode for FP8. Forcing the CUDA-core kernel (`SGLANG_FLASHINFER_USE_TENSOR_CORE=false`, mini-SGLang's choice for this model) reproduced mini-SGLang v1's collapse byte for byte.
5. **Sat image:** applied the ±448 clamp to SGLang's `set_kv_buffer` and reran both decode kernels. Both were clean.
6. Drafted an upstream comment (`UPSTREAM_COMMENT_DRAFT.md` (private draft), not posted). Experiment 8 later narrowed it.

## How to run

Run from the repository root, one container at a time.

```bash
cd /path/to/llm-inference-case-studies
A=projects/long-context-probe/4_sglang_fp8_kv_reproduction/app.py
uv run python -m py_compile $A
uv run modal run $A::main --dry-run
uv run modal run $A::main # 1. base image: auto, fp8_e4m3, fp8_e5m2
uv run modal run projects/long-context-probe/4_sglang_fp8_kv_reproduction/cast_probe.py # 2. torch cast behavior
uv run modal run $A::main --image probe --variants fp8_e4m3 # 3. log layer-0 K range before the cast
uv run modal run $A::main --image cudacore --variants fp8_e4m3_cudacore # 4. force CUDA-core decode
uv run modal run $A::main --image sat --variants fp8_e4m3_cudacore,fp8_e4m3 # 5. clamp patch, both decode kernels
uv run modal app list
```

Each run is one container. The worst case is (5 min startup + 30 min execution + 1 min idle) × one L4, 4 CPU cores, and 16 GiB, about **$0.67** at list prices; the runner reserves a conservative **$0.81**. Results go to `results/long-context-probe/sglang-fp8-kv-reproduction/<run-id>.json`, including the `KV Cache is allocated` and scale-warning log lines.

## Results

See [FINDINGS.md](FINDINGS.md).
