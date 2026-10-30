# Long-context inference study

This project asks where a single long prompt spends time in pinned mini-SGLang, whether time changes with context length, and whether FP8 KV storage is a feasible follow-up. It has three named experiments, each with its own folder, runbook, code, and result directory. All timings are **offline** `LLM.generate` timings, not client serving latency.

## Read and run in this order

| Experiment | Purpose | Status | Start here |
| --- | --- | --- | --- |
| 1. Profiling baseline | Run 8,000 input / 1,000 output and locate expensive work | Four runs completed 2026-10-23 | [Profiling runbook](1_profile_baseline/README.md), [findings](1_profile_baseline/FINDINGS.md) |
| 2. Context-length validation | Hold output at 1,000; compare 1,024, 4,096, and 8,000 input tokens on the unmodified pin | Six unprofiled runs and three short traces completed | [Validation runbook](2_context_length_validation/README.md), [findings](2_context_length_validation/FINDINGS.md) |
| 3. FP8 KV-cache feasibility | Compare BF16 with FP8 (e4m3) KV storage: KV bytes/capacity, quality, whole-call time, and which kernels change | v1: unscaled FP8 overflows (layer-0 K up to 516 > 448) into NaN, invalid. v2 clamp: correct on fixtures, decode attention −42.3%, whole call −13.7% (2 vs 2) | [Feasibility runbook](3_fp8_kv_cache_feasibility/README.md), [prediction](3_fp8_kv_cache_feasibility/PREDICTION.md), [findings](3_fp8_kv_cache_feasibility/FINDINGS.md) |
| 4. Full-SGLang FP8 reproduction | Does SGLang 0.5.9's FP8 KV path have the same overflow? | Same NaN bytes are written (layer-0 K up to 516); default output correct only because FP8 forces tensor-core decode; forcing CUDA-core decode reproduces mini-SGLang v1's failure byte-for-byte; the saturating clamp fixes it in SGLang; PR-comment draft ready (not posted) | [README](4_sglang_fp8_kv_reproduction/README.md), [findings](4_sglang_fp8_kv_reproduction/FINDINGS.md) |
| 5. FP8 capacity on W2 benchmark | Does doubling KV token capacity raise offline throughput on the sprint baseline workload? | +62.6% (1,818 → 2,956 tok/s, 2 vs 2): +47.6% from cheaper attention at identical batching, +10.5% from 2× concurrency; arithmetic accuracy 90.0% vs 91.0% BF16 (within noise) | [README and findings](5_fp8_kv_capacity_w2/README.md) |
| 6. FP8 break-even map | When does FP8 KV pay off across batch × context? | Gain from −1% (1 × 256) to +49% (32 × 4,096); step time = 5.84 ms + 4.02 ms/GB BF16 KV (R² 1.000); break-even about 616 tokens in flight | [README](6_fp8_break_even/README.md), [findings](6_fp8_break_even/FINDINGS.md) |
| 7. FP8 long-context retrieval | Does FP8 lose facts in 2k–8k contexts? | No FP8 loss (83 vs 84 of 90 at equal pool size); first-run "FP8 90/90 vs BF16 84/90" was an engine artifact (7b ablation) | [README](7_fp8_long_context_retrieval/README.md), [findings](7_fp8_long_context_retrieval/FINDINGS.md) |
| 8. SGLang FP8 upstream check | Is the MHA overflow live on SGLang main? | No: eager e4m3 casts saturate from torch 2.13 (main pins 2.14.1); the bug applies to releases with torch ≤ 2.12 (0.5.9) | [README](8_sglang_fp8_upstream_check/README.md), [findings](8_sglang_fp8_upstream_check/FINDINGS.md) |
| 9. mini-SGLang prefix-cache use-after-free | What caused experiment 7's failures? | Confirmed bug in `CacheManager.cache_req`: failing-then-passing regression test, detector shows all 6 failures were victims, 7-line fix gives 90/90 | [README](9_minisgl_prefix_cache_uaf/README.md), [findings](9_minisgl_prefix_cache_uaf/FINDINGS.md) |
| 10. Prefill/decode interference | Do long prefills stall in-flight decodes, and does chunk size help? | 4 × 8,000-token prefills freeze 32 decoding requests for about 1.2 s (vs 10 ms steps), completion +55%; chunk size does not help under prefill-first scheduling | [README](10_prefill_decode_interference/README.md), [findings](10_prefill_decode_interference/FINDINGS.md) |

The initial trace suggests decode attention and matrix-vector kernels account for much of the recorded GPU work. Experiment 2 found a longer whole-call time at each larger starting context; its matched traces show decode-attention kernel time rising with context while the main matrix-vector work stays nearly flat. This supports further KV attention investigation but does not prove an FP8 speedup or kernel-fusion opportunity. On 2026-10-23 the user chose to run experiment 3 early as an exploratory tweak, before the mini-SGLang map and CPU toy engine; it is not the W5 tweak selection in the canonical study plan (private study notes).

## File and artifact layout

```text
projects/long-context-probe/
  README.md                         this index and execution order
  1_profile_baseline/
    README.md                       commands and expected output for the completed probe
    app.py                          smoke, short profile, full, full-profile
    analyze_trace.py                streaming Chrome-trace event summary
    decode_step_trend.py            inferred per-step attention/GEMV trend
    FINDINGS.md                     observed results and limits
  2_context_length_validation/
    README.md                       six unprofiled runs and three short traces
    app.py                          pinned mini-SGLang measurement/trace runner
    table.py                        reads validation run records
    FINDINGS.md                     six-run timing result and next diagnostic
  3_fp8_kv_cache_feasibility/
    README.md                       what the patch changes, how Modal pins it, BF16–FP8 commands
    fp8_kv_cache.patch              25-line opt-in FP8 KV diff applied to the pinned base in the image
    app.py                          fresh BF16 control and patched FP8 runner with quality fixtures and traces
    table.py                        reads run records and compares greedy quality fixtures
    compare_traces.py               BF16 vs FP8 per-step and per-kernel trace comparison
    PREDICTION.md                   complete and commit before FP8 GPU work
    FINDINGS.md                     v1 failure, v2 fix, solutions, SGLang source notes, next steps
  4_sglang_fp8_kv_reproduction/
    README.md, FINDINGS.md          full-SGLang 0.5.9 FP8 reproduction and kernel-choice explanation
    app.py                          official SGLang image; base/sat/probe/cudacore variants
    cast_probe.py                   torch CUDA BF16->e4m3 saturation check
  5_fp8_kv_capacity_w2/            README, FINDINGS, run.sh, table.py (runner: exp 3 app.py --stage bench)
  6_fp8_break_even/                README, FINDINGS, run.sh, table.py (--stage sweep; bandwidth fit)
  7_fp8_long_context_retrieval/    README, FINDINGS, run.sh, table.py (--stage needle; 7b ablation)
  8_sglang_fp8_upstream_check/     README, FINDINGS, torch_cpu_bisect.sh, cast_probe_torch_versions.py
  9_minisgl_prefix_cache_uaf/      README, FINDINGS, run.sh, table.py, cache_req_uaf_fix.patch, regression_test_only.patch
  10_prefill_decode_interference/  README, FINDINGS, run.sh, table.py (--stage interference)

results/long-context-probe/         git-ignored raw artifacts
  <2026-10-23 profiling run-id>/    existing run.json and downloaded trace.json
  context-length-validation/        six unprofiled records and three trace records/files
  fp8-kv-cache-feasibility/          v1/v2 pilot and paired records, traces, diagnose/
  sglang-fp8-kv-reproduction/       experiment 4 records
  fp8-kv-capacity-w2/               experiment 5 records
  fp8-kv-accuracy-eval/             arithmetic eval records (exp 3)
  fp8-kv-break-even/                experiment 6 records
  fp8-kv-needle/                    experiments 7, 7b, 9 needle records
  minisgl-prefix-cache-uaf/         experiment 9 unit-test records
  minisgl-prefill-decode-interference/  experiment 10 records
```

The completed profiling results retain their original `results/long-context-probe/<run-id>/` locations; no raw artifact was renamed. Code and runbooks moved into the named folders. Each later app uses a distinct Modal app name and result directory, so the billing export and artifacts can be grouped by experiment. There is no new numbered study schedule here; the three runbooks describe how to execute their own experiment.

## Shared decisions and budget

All three experiments use mini-SGLang baseline commit `9a91cfafe754aa85daee49998176275667eb58f2`, the existing Qwen3-0.6B model Volume (expected revision `c1899de289a04d12100db370d81485cdf75e47ca`), BF16 model weights/activations, one Modal L4, and one GPU container at a time. The model snapshot is not independently hashed by these runners. Experiment 3 would change **only KV storage precision** in a fork at a full commit SHA; no fork source or upstream clone is stored in this repository.

At [Modal's listed prices checked 2026-10-23](https://modal.com/pricing), the configured one L4, 2 CPU cores, and 8 GiB memory cost about $0.016 per billable minute. The planned 5-minute startup, 15-minute execution, and 1-minute idle limit reserve **$0.34 per run**. Six unprofiled validation runs reserve **$2.04**; three short traces reserve another **$1.02**. An optional FP8 pilot and four paired runs reserve **$1.70** if the feasibility gate passes. The complete optional continuation reserves **$4.76**, subject to the shared sprint budget and current Usage. The first four profiling runs actually billed **$0.11884565** as an app-level total; see the [findings](1_profile_baseline/FINDINGS.md).

Before any paid command, read that experiment's cost gate, check `uv run modal app list`, reconcile Usage, and write the reservation on the study-plan status board. Afterward, record actual spend and cleanup. Six unprofiled context-length runs and three short traces completed on 2026-10-23. The three trace files are downloaded locally, all trace apps are stopped, and their refreshed posted charges total $0.02450739. No FP8 run has been recorded.

## Follow-up options

Status as of 2026-10-25. Earlier options: (1) the break-even map is done as experiment 6; (2) the upstream correctness contribution was resolved by experiment 8 (SGLang main is unaffected) and redirected to experiment 9's mini-SGLang bug; (3) long-context quality is done as experiment 7 (no FP8 gap, so v3 per-layer scales are not needed).

Next candidates, ordered by value:

1. **Full-SGLang source check, if pursuing an upstream contribution:** compare its prefix-cache page ownership path with experiment 9. Only pursue a full-SGLang fix if a current bug is independently reproduced there. The user decided on 2026-10-25 not to open a mini-SGLang PR; its fix and regression test remain local study artifacts.
2. **Serving-track measurement, after the data contracts in `docs/PROJECT_REFERENCE.md` exist** (it belongs to the SGLang serving case study, M1–M3, not this offline probe):
   - **S1, FP8 KV serving decision:** at a fixed request rate on SGLang, compare BF16 vs FP8 KV on client time to first token, time between tokens, completion latency, failures, and goodput. Use experiment 6's break-even estimate (about 616 tokens in flight) to choose a workload on each side of the break-even.
   - **S2, short/long interference online:** short-request latency with and without concurrent long prompts at a fixed arrival rate, recording queue and admission state where SGLang exposes it. This is the online counterpart of experiment 10.
3. **Request-order sensitivity (only if needed):** the same 90-prompt batch in a shuffled order on the fixed engine, to confirm that outputs are order-stable once the use-after-free is fixed.

Per the study plan (private study notes), the W3 mini-SGLang map and the user's CPU toy engine remain the core sprint work. Experiment 9 maps directly onto the radix cache and scheduler sections of that map.
