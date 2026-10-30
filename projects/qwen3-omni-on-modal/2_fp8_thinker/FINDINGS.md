# Experiment 2 findings

## Step 2a: FP8 smoke and gate

### What ran

Run `20261028T232413Z-73d3b038-fp8-smoke`: one Modal `H100!` (H100 80GB HBM3, driver 610.57.04), SGLang-Omni `305273a6`. Three text-only servers were started in sequence in one container, each with the radix cache off and `--thinker.engine.mem_fraction_static 0.784`:

- **B:** FP8 weights, BF16 KV.
- **C:** FP8 weights, FP8 e4m3 KV.
- **A:** BF16 weights and KV. This is the reference.

The FP8 weights are `marksverdhei/Qwen3-Omni-30B-A3B-FP8` @ `042db39a`, a **community FP8 conversion**, not an official Qwen release. Per condition:
- the 8k prompt from experiment 1 for 100 greedy tokens via `/generate` (token ids and logprobs);
- 3 free-generation prompts for 128 tokens;
- one streamed 8k/1k request.

Billed $0.62173545; app stopped. Account total after 2a: $6.79813925 of $50. Reproduce the numbers with `table.py`.

### Gate: both FP8 conditions pass

| Check | B | C |
| --- | --- | --- |
| CUDA BF16 → e4m3 cast (516 → 448, −516 → −448) | saturates | saturates |
| Non-finite logprobs (all 4 `/generate` requests) | 0 | 0 |
| 8k prompt, first 100 greedy tokens vs A | **100/100 identical** | **100/100 identical** |
| Streamed 8k/1k request valid (1,000 tokens, stops on length) | yes | yes |
| FP8 KV pool in the server log | n/a | `dtype: torch.float8_e4m3fn` |

The free-generation outputs diverge from A at tokens 72, 4, and 128 (none) for B, and at tokens 23, 30, and 30 for C. Divergence is expected with a different numeric path. All six outputs are coherent and correct when read: the same KV-cache explanation, the same four prefill/decode differences, and a working iterative Fibonacci function. This is a smoke check, not a quality measurement; the retrieval set in 2b is the quality judge. The copy-task gate is weak evidence for the reason given in the README.

### Speed and memory (n = 1 per condition; preview, not the 2b measurement)

| Condition | Weight load | KV dtype | **KV pool (tokens)** | TTFT, 8k prompt | **Decode, 8k/1k** |
| --- | ---: | --- | ---: | ---: | ---: |
| A: BF16 | 79.9 s (56.9 GB) | BF16 | **25,161** | 177.3 ms | **4.455 ms/token** |
| B: FP8 weights | 68.2 s (29.2 GB) | BF16 | **327,945** (13×) | 173.6 ms | **6.525 ms/token (+46%)** |
| C: FP8 + FP8 KV | 11.5 s (page cache warm) | FP8 e4m3 | **655,891** (26×) | 166.5 ms | **6.661 ms/token (+50%)** |

A's 4.455 ms/token matches experiment 1 (4.449–4.456), so the reference is stable across containers.

**Device copy bandwidth, measured:** 3,025 GB/s for a 2 GiB BF16 device-to-device copy (read + write), 90% of the 3.35 TB/s datasheet peak. Experiment 1's "% of peak" figures are therefore about 10% conservative. The measured BF16 floor at 8k is 2.07 × 3.35 / 3.03 ≈ 2.29 ms/token.

### What this shows

1. **FP8 is slower at batch 1 on this stack, by 46–50%.** That refutes the agent's revised expectation of 3.4–3.9 ms/token. FP8 halves the weight bytes, yet each decode step takes about 2.1 ms longer. So the FP8 kernels lose more time than the halved bytes save. The server log names the FP8 path:
 - **CUTLASS** for the MoE experts (`moe_runner_backend=cutlass`, which SGLang-Omni chooses automatically for native block-FP8 checkpoints).
 - **Triton** for the dense FP8 GEMMs (`fp8_gemm_backend=triton`). Upstream's comment says it chose Triton over DeepGEMM for 3–6% more throughput at concurrency 1 and 16; the comparison was against DeepGEMM, not BF16.
 - SGLang warns at startup: `Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found … N=2048,K=4096,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json` (and other shapes). So the dense FP8 GEMMs run with untuned default tiles.

 Which kernels lose the time is **not yet measured**: 2a has no trace.
2. **FP8 is a large capacity win.** At the same memory fraction, FP8 weights free 27.7 GB. The KV pool grows from 25,161 to 327,945 tokens (BF16 KV) or 655,891 tokens (FP8 KV). That is about 36 or 72 concurrent 9,000-token requests instead of 2. Under load, the per-token penalty may be repaid by batching; experiment 4 measures this.
3. **FP8 KV reaches the thinker and stores e4m3**, and the CUDA cast saturates. The torch ≤ 2.12 overflow from the old project does not occur. At batch 1, C is 0.14 ms/token slower than B; the cost of FP8 attention is not yet attributed.
4. **TTFT is unchanged** (167–177 ms): an 8k prefill is compute-bound, and FP8 does not change it here.
5. **Weight load scales with bytes, with the same caveats as experiment 1.** FP8 loaded 29.2 GB in 68.2 s (0.43 GB/s) cold from the Volume, and in 11.5 s when the same files were in the container's page cache (C after B). A loaded 56.9 GB at 0.71 GB/s. Cold loading stays slow and variable.

### Go / no-go for 2b

**Go, with a redesigned 2b.** Both conditions pass the gate. Measuring A vs B vs C again at batch 1 would only confirm the slowdown. The useful question is now:

> **Why is FP8 decode 46% slower than BF16 at batch 1, and does a backend or kernel-config choice recover it?**

Redesigned 2b, run in the order below. FP8 reloads take 11.5 s once the files are in page cache, so several sessions fit one container.

1. **B, traced.** Step anatomy with `analyze_trace.py` under both overlap rules, to show which categories grow over BF16 and by how much.
2. **B with `--thinker.engine.fp8_gemm_runner_backend deep_gemm`**, i.e. DeepGEMM for the dense layers. Check the name of the option value in the CPU check first.
3. **B with `--thinker.engine.moe_runner_backend triton`**, i.e. Triton fused MoE with FP8 instead of CUTLASS.
4. **A, again.** This is a drift check.

Decode ms/token is measured on 5 untraced 8k/1k requests per session, and each session also gets one trace. The retrieval quality set runs on A and on the fastest FP8 session. Cost: one container at the 2b row's 42-minute ceiling ($3.39), plus one repeat container for the winning configuration. That is less than the old 3-container 2b ($10.16).

If no backend choice brings FP8 to at least BF16 speed at batch 1, the customer recommendation becomes: **FP8 for throughput and capacity under load (experiment 4), BF16 for single-stream latency.** The missing tuned W8A8 block-FP8 config for these H100 shapes is then a concrete upstream question, which can be tested by running SGLang's FP8 GEMM tuning script.

### Limits

- n = 1 per condition for speed, in one container. The 46–50% gap is far larger than experiment 1's 0.3% container-to-container spread, but the 2b sessions repeat it.
- Smoke-level quality evidence only; the copy-task gate cannot detect small quality losses.
- The FP8 checkpoint is a community conversion; its quality is not Qwen's claim.
- FP8 uses `mem_fraction_static=0.784` (experiment 1's BF16 value), not upstream's colocated FP8 YAML. The KV pool sizes above depend on that choice.

## Step 2b-1: why FP8 is slower, by backend (/29)

### What ran

Run `20261028T234606Z-1cd46911-fp8-backends` (`app.py --stage fp8-backends`): one `H100!`, with four text-only server sessions in sequence. All use `mem_fraction_static` 0.784, radix cache off, and the 8k prompt from experiment 1. Each session: 1 warmup + 5 measured streamed 8k/1k requests. The three FP8 sessions also have 1 torch-traced request each. All 24 requests (and the 3 traced ones) are valid. Billed $1.11362538 + $0.16983228 = **$1.28345766** (the run crossed UTC midnight); account total $8.10988537; app stopped.

The B_default and B_deepgemm traces were saved as raw 1.7 GB JSON, because the 30 s export wait ended before upstream's background gzip finished. They were complete. `slim_trace.py` (new) streams them into small files containing only kernel and runtime events, then the raw files were gzipped. The analyzer now finds decode steps by the most common kernel count per CUDA-graph launch (unchanged result on experiment 1's trace: 4.274 ms), and has FP8 categories: `fp8_quant`, plus CUTLASS grouped-GEMM and permute kernels under MoE.

### Results (decode ms/token at 8k context, untraced; median [range] of 5)

| Session | Dense FP8 GEMM | MoE runner | Decode | vs BF16 | TTFT (median) |
| --- | --- | --- | ---: | ---: | ---: |
| B_default (SGLang-Omni's automatic choice) | Triton (`_w8a8_block_fp8_matmul`) | CUTLASS grouped GEMM | 6.524 [6.523–6.527] | +46% | 166.5 ms |
| B_deepgemm | **DeepGEMM** (`sm90_fp8_gemm_1d2d`) | CUTLASS | 5.488 [5.475–5.490] | +23% | 156.6 ms |
| B_moe_triton | Triton | **Triton fused MoE (FP8)** | 5.087 [5.084–5.088] | +14% | 178.0 ms |
| A (BF16, drift check) | cuBLAS (BF16) | Triton fused MoE (BF16) | 4.461 [4.447–4.462] | — | 176.6 ms |

A matches experiment 1 and 2a (4.449–4.461), so there is no drift across three containers.

### Step anatomy from the traces (median ms per decode step; BF16 column is experiment 1 container 2, trace L)

| Category | BF16 | FP8 default | FP8 + DeepGEMM dense | FP8 + Triton MoE |
| --- | ---: | ---: | ---: | ---: |
| Kernels per step | 820 | **1,300** | 1,300 | 964 |
| Dense GEMM | 1.387 | **2.242** | **1.270** | 2.241 |
| MoE | 1.829 | **2.260** | 2.276 | **1.422** |
| FP8 activation quantization | — | 0.238 | 0.233 | 0.208 |
| Attention | 0.963 | 0.928 | 0.934 | 0.932 |
| Norm | 0.165 | 0.205 | 0.205 | 0.187 |
| Other | 0.097 | 0.091 | 0.092 | 0.090 |
| **GPU idle** | 0.108 | **0.760** | **0.661** | 0.114 |
| Step period (traced) | 4.546 | 6.721 | 5.674 | 5.197 |

### What this shows

1. **The default FP8 path loses in all three places.**
 - Dense FP8 GEMMs on Triton with untuned default tiles take 2.24 ms, against 1.39 ms for BF16 cuBLAS, even though they read half the bytes.
 - The CUTLASS MoE path takes 2.26 ms, against 1.83 ms for BF16 Triton fused MoE.
 - The CUTLASS path also leaves the GPU idle 0.66–0.76 ms per step, inside the CUDA graph (the graph span grows with it).

 The step goes from 820 to 1,300 kernels. Per layer, that adds four activation-quantization kernels and the CUTLASS MoE's setup and permute kernels: problem sizes, expert offsets, argument sorts, row shuffles, a fill, a copy, and the scatter-sum.
2. **Each part has a better backend at batch 1.**
 - **DeepGEMM** for dense cuts dense GEMM from 2.24 to 1.27 ms, 8% *below* BF16 cuBLAS.
 - **Triton fused MoE** with FP8 cuts MoE from 2.26 to 1.42 ms (22% below BF16) and removes the idle gap (0.11 ms, as in BF16).
3. **Untested prediction for the combination** (DeepGEMM dense + Triton MoE): about 5.087 − (2.241 − 1.270) ≈ **4.1 ms/token, about 8% faster than BF16**. The two changes act on different categories, so their savings should roughly add up. 2b-2 tests this.
4. **Why upstream chose the default, and what this does and does not contradict.** SGLang-Omni's policy comment says Triton dense GEMM gave "3 to 6 percent more throughput" than DeepGEMM "at c1 and c16 … over 5000 MMSU prompts, and no post-ready compiles". MMSU prompts are short, prefill-heavy requests with variable shapes, where DeepGEMM's per-shape JIT compiles hurt. This experiment is 8k-context decode with 1,000 output tokens, where decode is 96% of the time. Both can be true. **The right FP8 backend depends on the workload**, and the automatic choice is not the fastest for long-output decode at batch 1. No compile spike appeared on the first 8k request with DeepGEMM (TTFT 160.9 ms), but only one prompt length was tested.
5. **FP8 KV is not involved here** (all sessions use BF16 KV). Attention time is the same as BF16 within 0.04 ms.

### Limits

- Batch 1, one prompt length (8,000), one container per configuration (A repeated across three containers).
- The combined configuration is not yet measured, and quality is not yet measured beyond the 2a gate.
- The idle time in the CUTLASS sessions is measured but not explained (no CPU stacks; the gaps are inside the CUDA graph, so they are not host launch gaps).
- MMSU-style throughput (upstream's metric) is not measured here.

### Next: 2b-2 (the repeat container planned in the 2a go/no-go)

One container, $3.39 ceiling:
- **B_combined** (`--thinker.engine.fp8_gemm_runner_backend deep_gemm --thinker.engine.moe_runner_backend triton`): 1 warmup + 5 measured 8k/1k requests and 1 trace.
- **Quality set** on A and B_combined: 60 needle-retrieval items at about 2k, 4k, and 8k tokens (keys generated in-container from a fixed seed and stored in the record), plus the 3 free-generation prompts. The varied prompt lengths also exercise DeepGEMM's per-shape compiles; per-request TTFT is recorded.
- **8 concurrent 8k/1k requests** on B_combined and B_default (both have a 328k-token KV pool), comparing total output tok/s. This checks whether the batch-1 ranking holds when the batch grows; upstream's choice was measured at concurrency 16.

## Step 2b-2: the combined FP8 configuration

### What ran

Run `20261029T001009Z-20f301d7-fp8-combined` (`app.py --stage fp8-combined`): one `H100!`, three sessions, all with `mem_fraction_static` 0.784 and the radix cache off.

- **B_combined:** FP8 + `--thinker.engine.fp8_gemm_runner_backend deep_gemm --thinker.engine.moe_runner_backend triton`. 1 warmup, 5 measured 8k/1k requests, 1 traced, the 60-item needle set, 3 free-generation prompts, and 8 concurrent 8k/1k requests.
- **B_default:** warmup + the same 8 concurrent requests.
- **A (BF16):** warmup + the needle set and the free-generation prompts.

The needle items come from a fixed seed (1234): 20 each at 1,500, 3,000, and 6,000 words of the pinned text (about 2k, 4k, and 8k tokens), with the code sentence at a random depth. The answer keys and prompt hashes are in the record. Every request is valid. Billed **$0.83696987**; account total $8.94685524 of $50; app stopped.

### Results

| | A: BF16 | B_default: FP8, automatic backends | **B_combined: FP8, DeepGEMM dense + Triton MoE** |
| --- | ---: | ---: | ---: |
| Decode at 8k, batch 1 (ms/token, median of 5) | 4.461 (2b-1) | 6.524 (2b-1) | **4.122** [4.122–4.125] (**−7.6% vs BF16**) |
| TTFT at 8k (median) | 176.6 ms (2b-1) | 166.5 ms (2b-1) | 166.8 ms |
| Traced step | 4.546 | 6.721 | 4.220 |
| Kernels per step | 820 | 1,300 | 964 |
| Needle retrieval (2k / 4k / 8k) | **20 / 20 / 20** | not run | **20 / 20 / 20** |
| 8 concurrent 8k/1k: output tok/s | not run (KV pool fits 2) | 687 | **827 (+20%)** |
| KV pool (tokens) | 25,161 | 327,945 | 327,988 |

The agent's prediction from 2b-1 (about 4.1 ms/token) held: 4.122. Step anatomy of B_combined (median ms): dense GEMM 1.262, MoE 1.428, attention 0.942, FP8 quant 0.197, norm 0.202, other 0.091, idle 0.097. These are the dense number from B_deepgemm and the MoE number from B_moe_triton, so the two backend changes add up as expected.

**The cost upstream warned about is real.** On B_combined, needle requests 1 and 6 of the 2k group took 0.461 s and 0.981 s, against a 0.083 s median for that group. Every later request at every length was normal (4k max 0.122 s, 8k max 0.203 s). BF16 had no such outliers (max 0.100 s at 2k). This pattern fits DeepGEMM compiling kernels for new prefill shapes after the server is ready (SGLang-Omni sets `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`), which is the reason upstream gives for preferring Triton dense GEMMs. The server log has no compile line, so it is consistent with that cause but not proven. **Corrected in 2c:** the same two requests were slow (0.41 and 0.37 s) with Triton dense GEMMs and no DeepGEMM, so DeepGEMM is not the only cause; see step 2c.

**Quality:** 60/60 for both A and B_combined (Wilson 95% interval for 60/60: 94.0–100%). The test can detect a large loss, not a small one. The free-generation outputs are in the record.

### What this means

1. **FP8 can beat BF16 at batch 1 on this model, but not with the automatic backend choice.** The automatic FP8 path is 46% slower than BF16; changing two backend flags makes it 7.6% faster. Both are FP8 with the same weights and quality on this set. The difference is 1.0 ms of dense GEMM (untuned Triton vs DeepGEMM) and 0.8 ms of MoE plus 0.65 ms of idle (CUTLASS grouped GEMM vs Triton fused MoE).
2. **The ranking holds at 8 concurrent requests** (+20% tok/s over the automatic choice). It is not tested at 16 (upstream's measurement point) or on short prefill-heavy requests (upstream's MMSU workload).
3. **The trade-off is first-request latency for new prompt shapes** (up to about 1 s, once per shape), unless DeepGEMM is precompiled or warmed on the expected shapes. *(2c: about 0.4 s of this also appears without DeepGEMM.)* A customer with long outputs should take the combined flags and warm common prompt lengths after startup. A customer with short, varied, prefill-heavy requests may prefer upstream's default.
4. **FP8 still gives a 13× larger KV pool at the same memory fraction**, so in a capacity setting it wins twice: faster per token with the combined flags, and many more requests in flight. Experiment 4 measures that.

### Limits

- One container for the combined configuration; batch-1 decode spread over 5 requests is 0.003 ms. The A reference comes from 2b-1 (three earlier containers agree within 0.3%).
- The concurrency check is one closed-loop burst of 8 identical prompts; it is not a load test.
- The cause of the compile outliers is inferred from timing and upstream's documented behavior.
- The FP8 checkpoint is a community conversion.

### Possible upstream contribution

A focused issue for SGLang-Omni: for Qwen3-Omni FP8 on H100, long-output decode at batch 1, the automatic policy (`cutlass` MoE + `triton` dense) is 58% slower than `triton` MoE + `deep_gemm` dense (6.52 vs 4.12 ms/token), with traces attributing the gap. Plus the missing W8A8 block-FP8 tuning configs for these shapes. It should be framed as workload-dependent, citing the policy's own MMSU/c16 rationale, and should ask whether the policy should depend on output length or be documented. **Drafted :** UPSTREAM_ISSUE_DRAFT.md (private draft), after searching existing issues (#454, #583, #2040, #711, #797 are related; none covers it). Not posted.

## Step 2c: can tuned Triton dense GEMMs replace DeepGEMM?

### Why

DeepGEMM made FP8 faster than BF16 (2b-2), but it compiles kernels after startup. Upstream chose Triton dense GEMMs to avoid that, and the startup log shows Triton running these two shapes with untuned default tiles. If tuned Triton tiles close the gap, a customer gets the speed without the compile, and SGLang gets a config file it lacks: its H100 block-FP8 dense configs cover only DeepSeek shapes, and no open PR adds these.

### What ran

- **First attempt, failed** (`ap-npcmaVX4wLFcmIomtUWNKs`, about 19 GPU-minutes): SGLang v0.5.22's `tuning_block_wise_kernel.py`, run unmodified per (shape, M). M = 1 took 980.8 s for N=5120,K=2048 and 34.1 s for N=2048,K=4096. The M = 2 call recompiled the whole search space and hit its time limit; the runner did not catch the timeout, so no session ran. The cause: Triton compiles each kernel config again for each specialization of M (M = 1, M divisible by 16, other M), and the full search space has 1,280 configs at about 0.77 s each. The full upstream tuning cannot fit one 30-minute container.
- **Retry** `20261030T040749Z-5925c374-fp8-tune` (plan reserve row): the same script's `tune()` function, unmodified, with the search space filtered to `BLOCK_SIZE_M = 16` (256 configs). It ran for M = 1, 2, 4, 8, 16, 32, 64 on both shapes in 378 s. The config files add an M = 4096 entry with SGLang's own default tile, so prefill behaves as untuned. The server log confirms both files were loaded ("Using configuration from …"). Then one session, **B_triton_tuned**: FP8 with `triton` dense and `triton` MoE, `mem_fraction_static` 0.784, radix cache off. It ran the 60-item needle set first, then 1 warmup, 5 measured 8k/1k requests, 1 trace, and 8 concurrent requests. All requests are valid; 60/60 needles. About 17 min of container time (≈ $1.40 by wall clock); app stopped.

### Results

| FP8 thinker (same checkpoint, batch 1, 8k context) | Dense GEMM ms/step (trace) | Decode ms/token (median of 5) | vs BF16 4.461 | 8 concurrent, tok/s |
| --- | ---: | ---: | ---: | ---: |
| Automatic policy (CUTLASS MoE + Triton dense, untuned) | 2.242 | 6.524 | +46% | 687 |
| Triton MoE + Triton dense, untuned (2b-1) | 2.241 | 5.087 | +14% | not run |
| **Triton MoE + Triton dense, tuned M ≤ 64 (2c)** | **1.549** | **4.391** [4.391–4.396] | **−1.6%** | **750** |
| Triton MoE + DeepGEMM dense (2b-2) | 1.262 | 4.122 | −7.6% | 827 |
| BF16 reference (cuBLAS dense) | 1.387 | 4.461 | — | — |

Traced step for B_triton_tuned: 4.492 ms, 964 kernels, GPU busy 97.3%, MoE 1.408 ms, attention 0.924, FP8 quant 0.211. TTFT at 8k: 180 ms (DeepGEMM 167 ms).

### What this shows

1. **Tuning recovers about 70% of the dense-GEMM gap.** Dense time falls from 2.24 to 1.55 ms per step (0.69 of the 0.98 ms gap to DeepGEMM), and decode becomes 1.6% faster than BF16 instead of 14% slower. DeepGEMM remains 0.29 ms per step faster.
2. **For a customer, tuned Triton is the no-compile option.** It is 6.5% slower per token than DeepGEMM, 9% slower in the 8-concurrent burst, and has no DeepGEMM JIT. Startup still compiles Triton kernels unless the Triton cache is kept (see experiment 3).
3. **Correction to 2b-2: the 2k-needle outliers are not DeepGEMM-specific.** With no DeepGEMM in the path, needle requests 1 and 6 of the 2k group again took 0.410 and 0.366 s (median 0.093 s); every later request at 2k, 4k, and 8k was normal. 2b-2 saw the same positions at 0.461 and 0.981 s. BF16 never exceeded 0.10 s. The common factor is the FP8 path with Triton MoE: a first-of-shape compile in the FP8 Triton MoE or quantization kernels fits, but is not proven. DeepGEMM may add to it (0.98 vs 0.37 s at request 6). The 2b-2 text above is corrected accordingly.
4. **A config PR is possible, but not with this file.** These configs come from a filtered search (BLOCK_SIZE_M = 16 only, M ≤ 64) and carry a default tile for prefill. An upstream PR needs the full `tuning_block_wise_kernel.py` run for both shapes, which needs about 3 × 16 min of compiles per shape class on one H100, or a persisted Triton cache across containers. With the issue posted first, maintainers can say whether they want that.

### Limits

- One container, one session; the A and DeepGEMM rows come from earlier containers (BF16 reproduced within 0.3% across four).
- The filtered search space can only under-state what full tuning would reach.
- Quality: the 60/60 needle set only.
