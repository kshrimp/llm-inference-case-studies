# Qwen3-Omni on Modal: long-context anatomy, FP8, cold start, and capacity

This project rebuilds the long-context and omni probes on one real, actively maintained engine. The earlier work stays where it is: [long-context probe](../long-context-probe/README.md) (mini-SGLang, Qwen3-0.6B, L4) and [omni pipeline probe](../omni-pipeline-probe/README.md) (SGLang-Omni `8200f499`, Qwen3-Omni, H100). Their numbers are not baselines here, because the engine version changes.

**The customer question.** A team wants to serve Qwen3-Omni on Modal for long text inputs (about 8,000 prompt tokens, 1,000 output tokens). What does one H100 deliver, where does each decode step's time go, should they run FP8, how long does a container take from zero to first token, and how much traffic does one GPU carry within a latency target?

**Result:** [RECOMMENDATION.md](RECOMMENDATION.md). Every run folder has `SHA256SUMS` and `validation.json` (`validate.py`).

**Pass condition for the whole project:** every customer-facing number in the final recommendation traces to a valid raw record under `results/qwen3-omni-on-modal/`, and every planned cost ceiling can be recomputed with [`budget.py`](budget.py).

## Why this design

- **One engine, current pin.** SGLang-Omni main is 141 commits past the old `8200f499` pin and now pins `torch==2.14.1` and `sglang==0.5.22`. The earlier FP8 KV overflow came from non-saturating e4m3 casts in torch ≤ 2.12 ([long-context experiment 8](../long-context-probe/8_sglang_fp8_upstream_check/FINDINGS.md)); it is expected not to recur, and experiment 2's smoke checks it.
- **Same workload shape as before.** 8,000 input / 1,000 output text tokens on the thinker, so the mini-SGLang findings (decode attention grows with context, prefill stalls decode) can be tested on a 30B MoE.
- **A roofline as a bound, not a judge.** [`roofline.py`](roofline.py) computes a peak-bandwidth lower bound from the pinned config: 3.04 B active parameters per token, so BF16 decode cannot beat **2.07 ms/token** at 8k context on H100 SXM (FP8 weights: 1.16; FP8 weights + FP8 KV: 1.03). These are floors, not predicted latency. Where measured time goes is established only by traces and controlled comparisons.

## Experiments, in order

| # | Question | Target | Evidence | Status |
| --- | --- | --- | --- | --- |
| 1 | [Long-context anatomy](1_long_context_anatomy/README.md): where does an 8k/1k request's time go on the current pin? | Attribute ≥ 95% of median decode-step time from the trace; anything else is reported as unattributed | Torch trace (CUDA + CPU), 1k/1k control at matched step positions, untraced medians | **Done** ([findings](1_long_context_anatomy/FINDINGS.md)): 4.45 ms/token at 8k vs 2.07 floor; GPU 97.7% busy; MoE 40%, dense GEMM 31%, attention 21%; 100% attributed |
| 2 | [FP8 thinker](2_fp8_thinker/README.md): BF16 vs FP8 weights (and FP8 KV, if gated in) | Decode within 1.5× of its own floor; no quality loss outside BF16's interval | Paired runs; held-out quality set vs BF16 reference | **2a done** ([findings](2_fp8_thinker/FINDINGS.md)): B and C pass the gate (100/100 tokens match BF16 on the 8k prompt; FP8 KV pool confirmed), but **FP8 decodes 46–50% slower at batch 1** (6.53–6.66 vs 4.46 ms/token) while the KV pool grows 13–26×. **2b-1 done:** the automatic FP8 backends (Triton dense with untuned tiles, CUTLASS MoE with 0.7 ms idle per step; 1,300 vs 820 kernels per step) cause it; DeepGEMM dense → 5.49 and Triton MoE → 5.09 ms/token; combination predicted about 4.1. **2b-2 done:** DeepGEMM dense + Triton MoE = **4.122 ms/token (−7.6% vs BF16)**, 60/60 needle retrieval (same as BF16), +20% tok/s over the automatic FP8 choice at 8 concurrent; cost: up to 1 s once per new prompt shape (DeepGEMM compile). **2c done:** Triton dense tiles tuned for M ≤ 64 (filtered search; the full upstream search failed on time in the first attempt) → 4.391 ms/token (−1.6% vs BF16; DeepGEMM −7.6%); the 2k first-request outliers also occur without DeepGEMM (2b-2 attribution corrected) |
| 3 | [Cold start](3_cold_start/README.md): zero to first token on Modal, and which levers cut it | Under 30 s | Client-clock and container-clock spans reported separately; cache state recorded per run | **Done** ([findings](3_cold_start/FINDINGS.md)): FP8 recommended config 201–278 s client-observed (median 275); kernel caches on a Volume → 174–199 s (median 190, −31%); BF16 165–238 s. Prefill CUDA-graph capture with DeepGEMM compiles (101–144 s) is the largest phase. GPU memory snapshot creation failed (multi-process server). Both targets (≤ 90 s, < 30 s) missed |
| 4 | [Capacity planning](4_capacity/README.md): highest 8k/1k load one H100 serves within a frozen latency target | Find the passing rate to within one sweep step | Open-loop client records with frozen pass criteria | **4a done** ([findings](4_capacity/FINDINGS.md)): automatic BF16 memory caps at 2 concurrent 8k requests (318 tok/s); `mem_fraction_static 0.90` → 730 tok/s (2.3×); FP8 combined at 0.90 → ≥ 1,390 tok/s (4.4×, not saturated); 0.93 fails the 95% memory gate. **4b done:** within p95 TTFT ≤ 3 s and p95 TBT ≤ 60 ms, FP8 combined 0.90 carries **1.357 req/s** (≈ $0.99 per 1,000 requests), BF16 0.90 0.490 req/s, BF16 default none ≥ 0.21 req/s; TTFT is always the limiting criterion |

Experiments 3 and 4 cover the systems questions for the Modal FDE track. Experiment 2 is the precision experiment. Experiment 1 comes first because both later experiments are interpreted against its step anatomy. After experiment 1, the run order is 2 → 4 → 3 (see the revised plan).

## Revised plan after experiment 1

Experiment 1 and a review of its raw data ([findings](1_long_context_anatomy/FINDINGS.md), "Review checks") changed what the later experiments should test:

| Finding | Change |
| --- | --- |
| Batch-1 decode is GPU-bound (97.7% busy) at 4.45 ms/token. The expert GEMMs run at about 70% of peak bandwidth; the per-layer projections are latency-bound at about 46%; there are 820 kernels per step | Experiment 2's FP8 prediction is revised from 2.8–2.9 to **3.4–3.9 ms/token**. The tuned fused-MoE config is capped at about 5% and stays optional |
| The default KV pool holds 25,161 tokens (two concurrent 9k requests) while **14.1 GiB is never used** | **New step 4a:** raise `mem_fraction_static` and run a closed-loop concurrency scan before the open-loop sweep; the sweep's rates now come from 4a by a frozen rule |
| Startup is 146–282 s, and the spread is weight loading (0.32 vs 1.20 GB/s for the same 56.9 GB). Prefill CUDA-graph capture adds 30–34 s | Experiment 3 records load throughput in every run, replaces the "persisted caches" condition with a startup-tuned condition (caches + fewer prefill graph buckets, with the TTFT cost measured), and splits the target: < 30 s with a snapshot, ≤ 90 s without |
| Copy-task outputs are a weak quality probe | Experiment 2's pass decision rests on retrieval + free-generation prompts; it adds a measured HBM copy bandwidth |
| Bugs found: the cold-start cause was misattributed in FINDINGS (fixed); experiment 4 assumed `ignore_eos` (fixed to copy prompts); per-kernel MoE attribution depends on the overlap rule (caveat added; category totals are robust within 0.04 ms) | — |

**Order stays 2 → 4 → 3**, with 4 before 3. Capacity is now the most valuable customer result: it depends on experiment 2's FP8 pass, and the snapshot work in 3 is the riskiest. Experiment 3 also gains from 4a, because it uses the chosen memory fraction.

## Gap review and upstream check (no GPU)

**What the project can already answer:** where batch-1 decode time goes (exp 1), whether to run FP8 and with which backends (exp 2), and the closed-loop throughput ceiling per memory setting (4a). **What it cannot answer yet:** the two customer questions in the opening paragraph that need 4b and 3.

| Gap | Why it matters | Cost | Priority |
| --- | --- | --- | --- |
| 4b open-loop sweep | The customer question is "how much traffic within a latency target"; 4a gives only burst ceilings. Prompts in 4b vary in length (7,915–8,000 tokens), so DeepGEMM post-ready compiles can land in measured windows: record per-request TTFT and flag first-of-shape requests. Add the planned c40 closed-loop level first | 92 min ceiling, $7.42 (needs OK) | Required |
| Exp 3 cold start, **on the recommended config** | The README still plans BF16 conditions. The recommendation will be FP8 combined at 0.90, which reads 29 GB instead of 57 GB but adds DeepGEMM compiles. Run baseline and startup-tuned on FP8 combined, keep one BF16 baseline for comparison, then snapshot | ≈ $10.48 at current rows | Required |
| Customer recommendation | The project's pass condition; does not exist yet | $0 | Required |
| Run validation + checksums + archive | Pass condition says "valid raw record". The omni probe has `validate.py`, `SHA256SUMS`, and a Volume backup; this project's runs have only `run.json` | $0 (≈ cents for the backup) | Required before the recommendation |
| Tuned Triton block-FP8 dense configs for the thinker shapes | Tests whether upstream's default dense backend can match DeepGEMM (1.27 vs 2.24 ms/step) **without** post-ready compiles; resolves exp 2's trade-off and is the one upstream PR with no existing work (see below) | ≈ 30–40 min H100, ≈ $3 | Recommended extra |
| BF16 fused-MoE config for triton 3.8.0 | Upper bound about 5% (exp 1 review) | Same container | Optional |
| CUTLASS idle inside the graph; 2.1 s gap at c2 (4a) | Unexplained, but neither changes the recommendation | — | Skip |

**Upstream candidates, checked against current mains** (SGLang-Omni `558f140b`, SGLang `0f3f6198`, mini-SGLang `9a91cfaf`, at the checked commits):

| Finding | Upstream state | Action |
| --- | --- | --- |
| FP8 backend policy is 46% slower than BF16 at batch-1 long decode; triton MoE + deep_gemm dense is 7.6% faster (exp 2) | `sglang_omni/platforms/cuda.py` unchanged since our pin; no issue covers it (searched: fp8 cutlass, deep_gemm, moe_runner_backend, fp8 decode). Related: SGLang-Omni #454, #583, #2040, #711; SGLang #43913 (SM90 small-M block-FP8 MoE, open) | **Ready to post** (draft (private draft)); the user posts |
| No tuned W8A8 block-FP8 dense configs for the thinker shapes on H100 (e.g. `N=2048,K=4096`) | SGLang main has 19 H100 block-FP8 dense configs, all DeepSeek shapes (K = 7168 etc.); no open PR for Qwen3-30B-A3B shapes | **PR candidate** after the tuning run above, if tuned Triton closes most of the gap |
| FP8 MoE (`E=128,N=768`, block-FP8, H100, triton 3.8.0) uses default config | SGLang draft PR **#42804** (Qwen3-30B-A3B-FP8 shapes, opened, no reviews yet) | Do not duplicate. Optional comment with a decode-heavy data point only if we test that config |
| BF16 fused MoE `E=128,N=768` H100 has no triton 3.8.0 config | Only `triton_3_2_0` and `triton_3_7_1` exist on SGLang main; no PR found | Low value (≤ 5%); bundle with the config PR only if measured |
| Text-only thinker's automatic memory fraction leaves 14 GiB unused (4a) | No issue found. #1089 changes colocated speech budgets only. Our test is text-only; image/video encoders need the reserve | **Do not file.** A default change is not justified by text-only evidence; it stays customer advice |
| Colocated speech hard-codes `enable_partial_start=False` (omni exp 3: TTFA 340 → 226 ms when on) | Still hard-coded on main (`Qwen3OmniSpeechColocatedPipelineConfig`); **open PR #2800** "Cut colocated streaming TTFA with a 4-frame first window and partial start" | Do not duplicate. Optional supportive comment with our A/B/A numbers (measured on the older `8200f499` pin, stated as such) |
| Torch trace has no CPU operator events (exp 1 limit) | Open PR #1304 fixes this | None |
| mini-SGLang `cache_req` use-after-free (long-context exp 9) | **Already reported:** issue #168 (open) with fix PRs #142 and #154, both open | Do not file. It is an independent reproduction, not a first report |
| SGLang FP8 KV NaN cast (long-context exp 4/8) | PR #41732 closed unmerged; current torch saturates | Drop the comment draft |

## Step 0 results (CPU check)

Run `20261028T042746Z-1955d27a-cpu-check` (record in `results/qwen3-omni-on-modal/0_setup/`; billed $0.00941960, app stopped).

- **Stack at the pin:** checkout matches `305273a6`; Python 3.12.15, torch 2.14.1+cu130, sglang 0.5.22, sglang-omni 0.2.0, transformers 5.19.0, flashinfer-python 0.7.1, flash-attn-4 4.0.0b35, triton 3.8.0. `sgl-kernel` and `deep-gemm` are null: not installed under those distribution names (DeepGEMM availability is checked from the server log in step 2a).
- **Cast probe:** BF16 → e4m3 maps 516 → 448 and −516 → −448, so the cast saturates. This ran on CPU only; the CUDA cast is checked on the GPU in step 2a.
- **Configs resolve:** BF16 text-only (radix cache off), FP8 text-only from upstream's `qwen3_omni_fp8_colocated.yaml`, and FP8 + `--thinker.engine.kv_cache_dtype fp8_e4m3` all resolve with exit 0. **The CPU half of condition C's gate passes:** the override reaches the thinker's resolved engine config. Whether the running engine actually allocates an FP8 KV pool is checked from the server log in step 2a.
- **Confound found:** the FP8 config sets the thinker's memory fraction to 0.75, while the BF16 text-only default sets none. A and B would then get different KV pools. Experiment 2 fixes the thinker memory fraction to the same value for A, B, and C and records each condition's KV token capacity from the server log.
- **Weights:** BF16 is on the `omni-hf-cache` Volume. FP8 was added by run `20261028T043011Z-0f882bb1-download-fp8`: 24 files, 37.39 GB, every size and all 15 LFS SHA-256 hashes match the Hub at `042db39a`; 3 min 7 s; billed $0.03662514; app stopped. Step 0 total: $0.04604474.
- Benign warnings: `torchada` missing (MUSA platform only); a transformers notice about unrecognized `rope_parameters` keys (`interleaved`, `mrope_section`), to be watched in the GPU smoke for any effect on output.

## Shared decisions

| Choice | Decision | Why |
| --- | --- | --- |
| Engine | SGLang-Omni `305273a673c281f2d214e1bed457a4b5b8895740` (pinned main commit) | Current torch 2.14.1 / sglang 0.5.22 stack; re-read before the first build and record any change |
| BF16 model | `Qwen/Qwen3-Omni-30B-A3B-Instruct` @ `26291f79…` (already on the `omni` weights Volume) | Same weights as the omni probe |
| FP8 model | `marksverdhei/Qwen3-Omni-30B-A3B-FP8` @ `042db39a102171d5994b72ab717fade92e80061a`, labelled **community FP8 conversion** in every result (user decision, option B). Rejected: load-time FP8 quantization of the BF16 weights (option A), because upstream supports only native block-FP8 checkpoints for Qwen3-Omni (its CUTLASS MoE path raises an error without `weight_block_size`), it would still read 60 GB of BF16 at cold start, and no customer would deploy it | Upstream's own FP8 configs name this checkpoint, and its loading was fixed in PR #454 (inverse block scales; ignored-layer matching). Upstream publishes no BF16 side-by-side, which is what experiment 2 adds |
| FP8 KV cache | Conditional. No Qwen3-Omni code or config sets `kv_cache_dtype` (other models do: MiniCPM-o, Qwen3-TTS, PRs #1128, #2917), so whether SGLang's generic `--kv-cache-dtype fp8_e4m3` reaches the thinker is unknown. Condition C runs only if the CPU config check shows it in the resolved thinker args **and** the 2a smoke passes the correctness gate | Avoids designing around a path that may not exist |
| GPU | One Modal `H100!` (SXM), one container at a time | Same GPU class as the omni probe; the roofline assumes SXM HBM3 |
| Metric classes | Kept separate: in-engine step and kernel time (exp 1–2), client-observed latency (exp 4), container lifecycle time (exp 3) | Project rule: never mix offline, serving, and scheduler measurements |

## Budget

Approved total: **$50**. Spent before this project: $4.647829 (last omni total). Run `python budget.py` for the table below; every ceiling is containers × (startup timeout + run timeout + idle) × the per-second price the omni probe used (read before the run). One GPU container (H100 + 8 cores + 64 GiB) is **$4.838 per hour**. Re-read the price before the first GPU launch.

| Step | Containers × minutes | Ceiling |
| --- | --- | ---: |
| 0. CPU image build + FP8 weight download | 2 × 66 (CPU) | $0.56 |
| 1. Anatomy | 2 × 42 | $6.77 |
| 2a. FP8 smoke and FP8 KV gate | 1 × 27 | $2.18 |
| 2b. FP8 comparison | 3 × 42 | $10.16 |
| 3. Cold start (3 conditions × 3 cold containers) | 9 × 12 | $8.71 |
| 3s. Snapshot creation | 1 × 22 | $1.77 |
| 4a. Memory fraction + closed-loop scan (added after exp 1) | 1 × 42 | $3.39 |
| 4b. Capacity open-loop sweep | 1 × 87 | $7.02 |
| Reserve (one 42-minute retry) | 1 × 42 | $3.39 |
| **Plan worst case** | | **$43.94** |

Steps 0 and 1 actually cost $1.52 (ceilings $7.33), so the worst-case account total with every remaining row at its ceiling is **$42.78 of $50** (`python budget.py`, last line). A failed step is still re-planned rather than re-run beyond the one reserve. The runner enforces each timeout, so no container can bill past its row.

Before each paid launch: write the worst-case line in the private study-plan cost log, check `uv run modal app list`, and reconcile with `modal billing report --json`. After it, record the actual spend and confirm every app is stopped.
