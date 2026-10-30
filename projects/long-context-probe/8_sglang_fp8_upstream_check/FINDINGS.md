# Experiment 8 findings: PyTorch fixed the eager overflow in 2.13, so SGLang main is not affected

**Result: the NaN failure from experiments 3–4 depends on the PyTorch version.** PyTorch's BF16 → `float8_e4m3fn` cast returns **NaN** for finite values above about 464 through **2.12.0**, and **saturates to ±448 from 2.13.0** on both CPU and CUDA. SGLang main pins torch **2.14.1**, so its unclamped `MHATokenToKVPool.set_kv_buffer` is safe at runtime. The reproduced bug affects SGLang releases that ship torch ≤ 2.12, such as **0.5.9 (torch 2.9.1)**, the version this project pins. There is no strong case for a new SGLang PR. The useful upstream contribution is narrower: telling PR #41732's author that its Python-path changes are runtime no-ops on main's torch, while its CUDA intrinsic change does matter.

## 1. Cast behavior by torch version

| torch | Where | 450 | 464 | 470 | 516 | −516 | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2.9.1+cu129 | CPU, CUDA (contiguous, strided, transposed, fp32 source) | 448 | — | **NaN** | **NaN** | **NaN** | [exp 4 `cast_probe.json`](../4_sglang_fp8_kv_reproduction/FINDINGS.md) |
| 2.10.0 | CPU | — | — | NaN | NaN | — | CPU bisect |
| 2.12.0 | CPU | — | — | NaN | NaN | — | CPU bisect |
| 2.12.0+cu130 | CPU and CUDA (contiguous, strided, fp32 source), L4 | 448 | 448 | **NaN** | **NaN** | **NaN** | `cast_probe_torch_versions.json` |
| 2.13.0 | CPU | — | — | **448** | **448** | — | CPU bisect |
| 2.14.0 | CPU | — | — | 448 | 448 | — | CPU bisect |
| 2.14.1+cpu | CPU | 448 | 448 | 448 | 448 | −448 | CPU check |
| 2.14.1+cu130 | CPU and CUDA (contiguous, strided, fp32 source), L4 | 448 | 448 | **448** | **448** | **−448** | `cast_probe_torch_versions.json` |

The change lies between 2.12.0 and 2.13.0. The exact upstream PyTorch commit was not identified.

## 2. SGLang main (source at `773cb938`)

- `MHATokenToKVPool.set_kv_buffer` still divides by an optional scale and then calls plain `cache_k.to(self.dtype)`, with no explicit clamp.
- `should_use_tensor_core` still forces tensor-core decode for FP8 KV and still honors `SGLANG_FLASHINFER_USE_TENSOR_CORE`.
- `saturate_to_fp8_range`'s docstring says "`float8_e4m3fn` saturates to its max finite value", and that only `float8_e4m3fnuz` returns NaN on overflow. That is **true for torch ≥ 2.13** (main pins `torch==2.14.1`) and **false for torch ≤ 2.12**, including SGLang 0.5.9's 2.9.1.
- Net: on main, the regular MHA writer saturates through PyTorch itself, and only the explicitly saturated prefix-valid path matters for ROCm's e4m3fnuz.

## 3. PR #41732 (checked /25)

- Open, not a draft, **no reviews**, 7 failing CI jobs (NPU and AMD suites plus their "finish" summary jobs), with no update since the check. Its own GPU test file had not been run on the branch, per its description.
- It adds `to_fp8_satfinite` (clamp to ±`finfo.max`, then cast) and uses it in the MLA Python writers. It also switches a CUDA JIT kernel (`set_mla_kv_concat_q.cuh`) from `__NV_NOSAT` to `__NV_SATFINITE`.
- **On main's torch 2.14.1, the Python-path clamps change nothing at runtime**, because eager casts already saturate. They still guard older or swapped torch versions. **The CUDA intrinsic change is real**: `__nv_cvt_float2_to_fp8x2(..., __NV_NOSAT, ...)` does not depend on the torch version. This is inferred from the API semantics; the kernel was not run here.
- It does not touch `MHATokenToKVPool.set_kv_buffer`. Per section 2, that matters only on torch ≤ 2.12.

## Upstream position (revised)

- **Do not open an SGLang MHA PR** on the strength of the 0.5.9 reproduction. Current main is unaffected at runtime.
- **Optional, useful comment on #41732 (the user's decision):** report that eager e4m3fn casts saturate since torch 2.13, so the PR's Python changes are defense-in-depth on main, while the JIT `__NV_SATFINITE` change is the functional fix. Also report a dense-model reproduction on 0.5.9 (torch 2.9.1) showing what the old behavior does in practice: NaN collapse under CUDA-core decode, masked by tensor-core decode by default. The draft is rewritten accordingly in UPSTREAM_COMMENT_DRAFT.md (private draft).
- **mini-SGLang:** its pinned image installs torch 2.9.1, so the v1 NaN result stands for that stack. With torch ≥ 2.13 the v1 patch would likely not have collapsed (untested). The saturating v2 cast remains correct on any torch.
- A stronger upstream candidate came out of experiment 7: a **mini-SGLang prefix-cache use-after-free**, investigated in [experiment 9](../9_minisgl_prefix_cache_uaf/README.md).

## Limits

- The bisect is CPU-only for 2.10, 2.13, and 2.14.0; CUDA was checked only at 2.9.1, 2.12.0, and 2.14.1.
- SGLang main was read, not run. No newer SGLang image was tested.

## Cost

The two CUDA probe containers (app `fp8-cast-probe-torch-versions`) cost $0.026299 actual, including two image builds. The CPU bisect ran locally at no cost.
