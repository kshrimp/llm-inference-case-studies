# FP8 KV-cache feasibility in mini-SGLang

**Status: v1 (2026-10-23) failed with NaN, from layer-0 K overflowing ±448. v2 (2026-10-24), a saturating cast, is correct on the fixtures, with decode attention −42.3% per step and whole call −13.7%; see [FINDINGS.md](FINDINGS.md).** Next (ranked in [FINDINGS.md](FINDINGS.md#next-steps-revised-2026-10-24)): reproduce in full SGLang, whose regular KV write path also lacks saturation in source; test FP8 capacity on the W2 throughput benchmark; defer v3 scales. This experiment asks whether storing KV in FP8 (e4m3) instead of BF16 reduces KV bytes and speeds up the 8,000-input / 1,000-output offline workload, and which kernels change. The user chose on 2026-10-23 to run it as an early, exploratory tweak. It follows the [profiling baseline](../1_profile_baseline/FINDINGS.md) and [context-length validation](../2_context_length_validation/FINDINGS.md). The prediction is in [PREDICTION.md](PREDICTION.md). All GPU work runs on Modal.

## Why we ran this

Experiments 1 and 2 showed that, for one long prompt, decode attention is one of the two largest GPU costs, and that it grows with context while the matrix-vector (GEMV) work stays flat. Decode attention is bound by memory bandwidth: every step re-reads the whole KV cache. Storing KV in 1-byte FP8 instead of 2-byte BF16 halves those bytes. So this was the cheapest tweak that targets the measured bottleneck directly. The goal was to find out whether it is (a) mechanically possible in pinned mini-SGLang, which has no FP8 KV option, (b) faster per step and per call, and (c) still correct. It also gave the user practice at writing a falsifiable prediction before a paid run.

## What we did

1. Wrote a 25-line opt-in patch ([`fp8_kv_cache.patch`](fp8_kv_cache.patch)) that stores K/V as `float8_e4m3fn` with scale 1.0 and has FlashInfer dequantize inside attention. Weights, activations, and queries stay BF16.
2. Built two Modal images from the same upstream commit and the same pinned packages: unmodified BF16 control and patched FP8.
3. **v1 (2026-10-23):** ran a 1,024-token FP8 pilot, then BF16, FP8, FP8, BF16 at 8,000 input / 1,000 output, with matched 128-output traces on repeat 1. FP8 was 43% faster in decode attention, but its output collapsed to token `0`.
4. Added a `diagnose` stage that scans the stored KV. It found layer-0 K values up to 516, above e4m3's ±448 range, which PyTorch 2.9.1's cast turns into NaN.
5. **v2 (2026-10-24):** wrote a new prediction, then added a clamp to ±448 before the cast ([`fp8_kv_cache_v2_saturate.patch`](fp8_kv_cache_v2_saturate.patch)) and repeated the pilot, diagnostics, paired runs, and trace comparison.
6. Ran a 200-problem arithmetic accuracy check (`--stage eval`) for BF16, v1, and v2, scored with [`score_eval.py`](score_eval.py).

This `app.py` is also the shared runner for experiments 5, 6, 7, 9, and 10 (stages `bench`, `sweep`, `needle`, `unittest`, `interference`).

## How to run

Run from the repository root, one container at a time, after the cost gate in section 2 below. Full sequence:

```bash
cd /path/to/llm-inference-case-studies
D=projects/long-context-probe/3_fp8_kv_cache_feasibility
R=results/long-context-probe/fp8-kv-cache-feasibility

# Local checks (no GPU)
uv run python -m py_compile $D/app.py $D/table.py $D/compare_traces.py $D/score_eval.py
uv run modal run $D/app.py::main --stage pilot --variant fp8 --input-length 1024 --dry-run

# v1: plain cast (reproduces the NaN failure)
uv run modal run $D/app.py::main --stage pilot --variant fp8 --input-length 1024 --repeat 1
uv run modal run $D/app.py::main --stage final --variant bf16 --input-length 8000 --repeat 1
uv run modal run $D/app.py::main --stage final --variant fp8  --input-length 8000 --repeat 1
uv run modal run $D/app.py::main --stage final --variant fp8  --input-length 8000 --repeat 2
uv run modal run $D/app.py::main --stage final --variant bf16 --input-length 8000 --repeat 2
uv run modal run $D/app.py::main --stage diagnose --variant bf16 --input-length 1024
uv run modal run $D/app.py::main --stage diagnose --variant fp8  --input-length 1024

# v2: saturating cast
uv run modal run $D/app.py::main --stage pilot --variant fp8_sat --input-length 1024 --repeat 1
uv run modal run $D/app.py::main --stage diagnose --variant fp8_sat --input-length 1024
uv run modal run $D/app.py::main --stage final --variant bf16    --input-length 8000 --repeat 1
uv run modal run $D/app.py::main --stage final --variant fp8_sat --input-length 8000 --repeat 1
uv run modal run $D/app.py::main --stage final --variant fp8_sat --input-length 8000 --repeat 2
uv run modal run $D/app.py::main --stage final --variant bf16    --input-length 8000 --repeat 2

# Arithmetic accuracy check (one container each)
uv run modal run $D/app.py::main --stage eval --variant bf16
uv run modal run $D/app.py::main --stage eval --variant fp8_sat
uv run modal run $D/app.py::main --stage eval --variant fp8

# Analysis (local); pass exact trace paths when more than one run-1 folder matches
uv run python $D/table.py
uv run python $D/compare_traces.py $R/*-final-bf16-8000-r1/trace.json $R/*-final-fp8_sat-8000-r1/trace.json
uv run python $D/score_eval.py
uv run modal app list   # expect no running long-context-fp8-kv-cache app
```

The sections below explain each step, what to expect, and when to stop.

## v2: saturating cast (`fp8_sat`)

v1 failed because PyTorch's FP8 cast turns values above about 464 into NaN, and layer-0 K reaches 516 ([findings](FINDINGS.md)). [`fp8_kv_cache_v2_saturate.patch`](fp8_kv_cache_v2_saturate.patch) is the full v1 diff plus a clamp to ±448 before the cast. Every command below accepts `--variant fp8_sat`; `--variant fp8` still reproduces v1. The v2 prediction is at the end of [PREDICTION.md](PREDICTION.md). v2 order: pilot, then both diagnostics, then BF16, fp8_sat, fp8_sat, BF16.

```bash
uv run modal run $D/app.py::main --stage pilot --variant fp8_sat --input-length 1024 --repeat 1
uv run modal run $D/app.py::main --stage diagnose --variant fp8_sat --input-length 1024
uv run python $D/compare_traces.py $R/*-final-bf16-8000-r1/trace.json $R/*-final-fp8_sat-8000-r1/trace.json
```

When more than one BF16 run 1 exists, pass the matched trace's exact path instead of a glob.

## What changes, and what does not

The model is unchanged. Qwen3-0.6B keeps BF16 weights, activations, and queries. Only the **KV cache storage** changes, and it lives in mini-SGLang, not in the model. The pinned engine has no FP8 KV option, so [`fp8_kv_cache.patch`](fp8_kv_cache.patch) adds one. It is 25 added lines across four files, with an opt-in `kv_cache_dtype="fp8_e4m3"` (default `"auto"` keeps BF16):

| File (pinned `9a91cfaf`) | Change | Why |
| --- | --- | --- |
| `engine/config.py` | New `kv_cache_dtype` field | Opt-in switch; BF16 stays the default |
| `engine/engine.py` | `_resolve_kv_dtype`; pool allocated with the KV dtype; `_determine_num_pages` uses its 1-byte size; backend told the query dtype; non-FlashInfer backends rejected | Allocation and page-capacity arithmetic |
| `kvcache/mha_pool.py` | Cast K/V to the cache dtype before `store_cache` | The store kernel is a raw byte copy that requires matching dtypes |
| `attention/fi.py` | Plan with `q_data_type=BF16`, `kv_data_type=FP8` | FlashInfer dequantizes FP8 KV inside its prefill and decode kernels |

The scale is 1.0: a direct cast with no calibration. This matches full SGLang's default when a checkpoint has no KV scales, and it is the main quality risk. [Full SGLang documents FP8 KV modes](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/quantized_kv_cache.mdx), but that does not validate this patch.

**How Modal pins the modified engine.** No fork or GitHub push is needed. Both images clone upstream at the exact base commit and install the same pinned dependency versions (`PINS` in [app.py](app.py)). Upstream lists only lower bounds such as `flashinfer-python>=0.5.3`, which would let the two images drift. The FP8 image also copies the patch from this folder and runs `git apply`. Each run records the HEAD commit, `git diff --stat`, the patch SHA-256, and package versions. The BF16 control is unmodified upstream, rebuilt with the same pins, so it is a fresh control and not experiment 2's image. The study clone with the same diff is `~/projects/upstream/mini-sglang-fp8-kv`, branch `fp8-kv-study`.

Review the patch before launching:

```bash
cat projects/long-context-probe/3_fp8_kv_cache_feasibility/fp8_kv_cache.patch
git -C ~/projects/upstream/mini-sglang-fp8-kv diff --stat
```

## What each run records

Each container takes one variant. It runs the warmup, then the **unprofiled timed call** (input / 1,000 output). It then runs **8 natural-language quality prompts** (greedy, 128 tokens). In `final` repeat 1 only, it then captures a **128-output Chrome trace**, which is returned gzipped and saved as `trace.json` in the run folder. Timing comes first, so quality prompts and profiling cannot warm the timed call. `run.json` holds the KV dtype, pool shape, allocated bytes, token capacity, bytes per token, logical request KV, output hash, quality-fixture tokens and text, and versions.

## 1. Local checks (no GPU)

```bash
cd /path/to/llm-inference-case-studies
D=projects/long-context-probe/3_fp8_kv_cache_feasibility
uv run python -m py_compile $D/app.py $D/table.py $D/compare_traces.py
uv run modal run $D/app.py::main --stage pilot --variant fp8 --input-length 1024 --dry-run
uv run modal app list
```

Expect `"gpu_launched": false`, `"patch_sha256": "34379d45…"`, the pins, and `"estimated_max_usd": 0.34`. No `long-context-fp8-kv-cache` app should be running.

## 2. Cost gate

Before **each** paid command, check Modal Usage and [current prices](https://modal.com/pricing). Then confirm the reservation on the study-plan status board (private study notes). At prices checked 2026-10-23, one container reserves **$0.34** (5 min startup + 15 min execution + 1 min idle). One pilot plus four paired runs reserve **$1.70**. Run one container at a time.

## 3. FP8 compatibility pilot (1 container)

```bash
uv run modal run $D/app.py::main --stage pilot --variant fp8 --input-length 1024 --repeat 1
uv run python $D/table.py
```

Expect `"status": "completed"`, 1000/1000 tokens, `float8_e4m3fn`, and about 2× the BF16 pool's token capacity (172,544 in experiment 2). The app fails if the KV dtype is wrong. Read the quality-fixture `text` in `run.json`. **Stop** if the text is garbage, the dtype or backend is rejected, or the output is short.

## 4. Paired measurement with traces (4 containers, in this order)

```bash
uv run modal run $D/app.py::main --stage final --variant bf16 --input-length 8000 --repeat 1
uv run modal run $D/app.py::main --stage final --variant fp8  --input-length 8000 --repeat 1
uv run modal run $D/app.py::main --stage final --variant fp8  --input-length 8000 --repeat 2
uv run modal run $D/app.py::main --stage final --variant bf16 --input-length 8000 --repeat 2
uv run python $D/table.py
```

Repeat 1 of each variant saves `trace.json`. Expect four completed rows. The quality section lists, for each prompt, the first token index where FP8 differs from BF16 (`None` means identical).

## 5. Which step improved: trace comparison

```bash
R=results/long-context-probe/fp8-kv-cache-feasibility
uv run python $D/compare_traces.py $R/*-final-bf16-8000-r1/trace.json $R/*-final-fp8-8000-r1/trace.json
```

This reports per-step medians (first 32 decode steps) for decode-attention kernel time, decode-step start interval, and main GEMV time. It also lists every kernel ranked by its BF16→FP8 time change. Expect decode attention (`BatchDecodeWithPagedKVCacheKernel`) to fall, GEMV to stay flat, and new small cast kernels (`elementwise`/copy) plus `store_kv_cache` to appear. Prefill attention (`BatchPrefillWithPagedKVCacheKernel`) also reads FP8 KV. The analyzer was checked against experiment 2: it reproduces 0.554/3.725 ms attention and 3.77 ms GEMV.

## 5b. KV range diagnostic (when outputs look wrong)

```bash
uv run modal run $D/app.py::main --stage diagnose --variant bf16 --input-length 1024
uv run modal run $D/app.py::main --stage diagnose --variant fp8 --input-length 1024
```

Each zeroes the pool, generates 32 tokens, and saves per-layer max |K|/|V|, the count of values above 448, and FP8 NaN bytes under `results/.../fp8-kv-cache-feasibility/diagnose/`.

## 6. Close and record

```bash
uv run modal app list
uv run modal billing report --start 2026-10-01 --end 2026-11-01 --json > report.json  # --csv wraps at 80 columns when piped
```

Expect no running `long-context-fp8-kv-cache` app. Record actual app cost on the status board. Append results to `PREDICTION.md` and write `FINDINGS.md`: timing pairs and spread, KV bytes and capacity, quality divergence, and the trace comparison. Report negative or inconclusive outcomes honestly. A single request cannot show higher concurrency capacity. A performance contribution to full SGLang or vLLM needs a separate, pinned reproduction there.
