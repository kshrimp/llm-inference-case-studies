# Experiment 1 findings: where an 8k / 1k request's time goes (BF16, one H100)

## What ran

SGLang-Omni `305273a6` (torch 2.14.1, sglang 0.5.22), `Qwen/Qwen3-Omni-30B-A3B-Instruct` @ `26291f79`, BF16, text-only server with the thinker radix cache off, one Modal `H100!` (H100 80GB HBM3, driver 610.57.04). Prompts are copy tasks ("Repeat this text exactly:") over Project Gutenberg #132 (SHA-256 `2301ae1c…`, the omni probe's pin), sized by the server's own token count: **L** = 8,000 prompt tokens (6,057 words), **S** = 1,150 (807 words). The chat API has no `ignore_eos`, so both texts are longer than the output cap, and every valid request stops on length at exactly 1,000 tokens. Batch 1, greedy, sequential.

| Container | Run | Result | Server ready |
| --- | --- | --- | ---: |
| 1 (failed) | `20261028T044516Z-7012ec81-anatomy` | Runner bug: the prompt-sizing search stopped when a probe was rejected as too long. No measurement. Fixed in `size_prompt` | 242.3 s |
| 1 | `20261028T045033Z-1b4384f1-anatomy` | Complete | 282.3 s |
| 2 | `20261028T050802Z-e11e8999-anatomy` | Complete, independent repeat | 146.2 s |

Each complete container: 2 warmups, L and S interleaved 5 times each (untraced), 1 torch-traced L and 1 traced S, 1 L under py-spy. **All 20 measured requests and all 6 extra requests are valid** (completed, `finish_reason=length`, 1,000 completion tokens, prompt counts exactly on target). Reproduce with `table.py` (client numbers) and `analyze_trace.py` (trace numbers); raw records are in `results/qwen3-omni-on-modal/1_anatomy/`.

## Results

**Client-observed (untraced, n = 5 per prompt per container; ms):**

| Container | Prompt | TTFT, median [range] | Decode ms/token, median [range] |
| --- | --- | --- | --- |
| 1 | L (8,000) | 177.6 [176.0–180.2] | 4.449 [4.445–4.455] |
| 1 | S (1,150) | 49.0 [48.8–49.3] | 4.177 [4.177–4.179] |
| 2 | L (8,000) | 180.0 [178.1–180.4] | 4.456 [4.453–4.458] |
| 2 | S (1,150) | 49.8 [48.9–50.9] | 4.188 [4.186–4.190] |

Decode ms/token = (completion − TTFT) / 999. The two containers agree within 0.3%.

**Decode-step anatomy from the traces (median over 999 steps; ms/step):**

| Category | L, c1 | L, c2 | S, c1 | S, c2 | Bandwidth floor (L) | Achieved fraction of peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MoE (fused MoE GEMMs, activation, sum-reduce, router, align) | 1.827 | 1.829 | 1.820 | 1.821 | 1.082 | 59% |
| Dense GEMM (QKV/O projections, router GEMM, LM head) | 1.385 | 1.387 | 1.383 | 1.384 | 0.734 | 53% |
| Attention (FA3 forward + combine, mRoPE, QK-norm) | 0.959 | 0.963 | 0.697 | 0.701 | 0.249 (KV reads) | see below |
| Norm | 0.164 | 0.165 | 0.163 | 0.164 | | |
| Other kernels | 0.096 | 0.097 | 0.097 | 0.097 | | |
| **GPU idle (no kernel running)** | **0.102** | **0.108** | **0.101** | **0.108** | | |
| **Step period** | **4.536** | **4.546** | **4.264** | **4.274** | **2.07 (total)** | 46% |

GPU busy share: 97.5–97.7% of every step period. Time attribution: kernels by category plus measured idle cover 100% of the median step; **unattributed = 0%**, so the ≥ 95% target is met. The torch profiler adds 2% (traced L 4.54 vs untraced 4.45 ms), well inside the 15% stop condition.

**Method note: why durations are not summed.** All kernels ran on one stream, yet per-kernel durations summed to 6.2 s against 4.58 s of busy time in trace L. Consecutive kernel records overlap, consistent with programmatic dependent launch on Hopper (the next kernel launches early and waits). `analyze_trace.py` therefore attributes each busy instant to the earliest-started running kernel. Decode steps are identified as the CUDA-graph launches that contain exactly 96 fused-MoE kernels (48 layers × gate_up and down); 1,000 were found in every trace.

## Analysis

1. **Decode is GPU-bound, not CPU-bound.** The GPU is idle only about 0.1 ms per 4.5 ms step. The CPU launches each decode graph about 2.6–2.9 ms before its first kernel runs, so it is a full step ahead (consistent with the one-step-lookahead async decode of upstream PR #908). The 2–3 ms idle gap per step seen on the old pin's speech server ([omni exp 3](../../omni-pipeline-probe/3_speech_output/FINDINGS.md)) does not appear on this text-only server.
2. **The excess over the roofline is kernel efficiency at batch 1.** The step takes 4.45 ms against a 2.07 ms bandwidth floor. MoE and dense GEMMs together take 3.21 ms against a 1.82 ms floor, so they run at 53–59% of peak bandwidth. Small-M GEMMs at batch 1 do not saturate HBM; this is where the time is.
3. **Context length costs only attention, and that increment is bandwidth-efficient.** L − S is +0.272 ms per step in the trace (container 1) and +0.26 to +0.30 ms in the client data, at every step range. Attention accounts for +0.262 ms of it; MoE, dense, and norm differ by ≤ 0.007 ms. The bytes alone predict 0.201 ms (6,850 more tokens × 96 KiB of KV at peak bandwidth), so the extra KV reads run at about 77% of peak. **But attention has a large fixed cost:** at 1,150 context it still takes 0.70 ms per step against a 0.05 ms KV floor, about 14 µs per layer. At this context length, attention is latency-bound, not bandwidth-bound.
4. **Prefill is small at this size.** TTFT is 178 ms for 8,000 tokens and 49 ms for 1,150, so prefill is 4% of an 8k/1k request; decode is 96%. Per-token decode time is what matters for this workload.
5. **A concrete improvement lead in the server log.** SGLang has no tuned fused-MoE Triton config for this shape on triton 3.8.0 (`E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3.json`, and the `_down` variant). It falls back to the triton 3.7.1 config and warns "Performance might be sub-optimal!" MoE is the largest category (40% of the step, 59% of peak bandwidth). Whether a config tuned for triton 3.8.0 helps is untested.

**Agent expectation check (not the user's):** decode was expected at 3.5–5 ms/token (measured 4.45, inside). The expectation that "fused MoE at M = 1 holds most of the excess" is partly right: MoE holds the largest share, but dense GEMMs are almost as far from their floor. The expectation that L − S > 0.2 ms held (0.26–0.30).

## What this means for experiments 2–4

- **FP8 (experiment 2).** FP8 halves the bytes of the two categories that hold 71% of the step. If FP8 kernels reach the same fraction of peak, B would save about (1.83 + 1.39) / 2 ≈ 1.6 ms per step, so about 2.8–2.9 ms/token. *Revised in the review below to 3.4–3.9 ms/token*, because most of the dense GEMM time is per-kernel latency, not bytes. FP8 KV (condition C) can save at most about half of the 0.25 ms KV floor at 8k, about 0.1–0.15 ms, because most attention time is fixed cost.
- **Cold start (experiment 3).** Server readiness took 242, 282, and 146 s in three containers. *Corrected in the review below:* the 136 s difference between the two complete containers is weight loading (176.9 vs 47.3 s for the same 56.9 GB), not the JIT cache. These are observations, not a controlled cold-start measurement.
- **Capacity (experiment 4).** The BF16 thinker's KV pool holds 25,161 tokens (2.30 GB at `mem_fraction_static` 0.784). That is only two concurrent 9,000-token requests at full length, so BF16 capacity at 8k/1k is limited by KV memory before compute. Experiment 4's rates must account for this. A larger memory fraction or FP8 KV changes it directly.

## Limits

- One model, one GPU type, batch 1, one prompt source, and a copy task. A copy task decodes like any other greedy text at this batch size, but the outputs are not representative of quality.
- The trace categories come from kernel-name matching (`analyze_trace.py`). "Dense GEMM" includes the LM head and the router GEMM.
- **py-spy did not run:** Modal's sandbox denied ptrace (`Permission denied (os error 13)`) for every process. CPU stacks are null with that reason. Because the GPU is 97.7% busy, CPU-side attribution was not needed for the 95% target.
- The torch trace recorded CUDA runtime calls but no CPU operator events, so the launch lead is measured from `cudaGraphLaunch` to the first kernel.
- Cost: three GPU containers (including the failed one) billed $0.38858301, $0.43991874, and $0.64267968 in app rows, $1.47118143 in total against a $6.77 ceiling; the account total after experiment 1 is $6.1764038.

## Review checks (no new GPU runs)

A second pass over the raw traces, server logs, and GPU samples, to test the claims above. Reproduce with `kernel_breakdown.py <trace>` and `startup_memory.py`.

**1. The category split is robust to the overlap rule; the per-kernel split inside MoE is not.** When overlapping time goes to the latest-started kernel instead of the earliest, category totals move by at most 0.04 ms (container 2 trace L, 40 sampled steps): MoE 1.829 → 1.819 ms, dense 1.387 → 1.347, attention 0.958 → 0.959, norm 0.164 → 0.213. But inside MoE, the earliest rule gives `fused_moe_kernel` 1.540 ms, while the latest rule gives it 0.084 ms and moves the time to `act_and_mul` (0.982) and `moe_sum_reduce` (0.561). Their records overlap the GEMM almost completely. **So the trace supports "MoE takes 1.82 ms" but cannot say how that splits between the expert GEMMs and the elementwise kernels.** Under the earliest rule, the 96 `fused_moe` calls move 3.62 GB in 1.54 ms, about 70% of peak bandwidth, and router, align, and the other MoE kernels add about 0.29 ms.

**2. Dense GEMM is two different stories.** The LM head (one 384×8-tile cuBLAS kernel, 0.62 GB) takes 0.215 ms per step, 87% of peak bandwidth. The 48 layers' projection GEMMs (three cuBLAS kernels per layer plus a split-K reduce, about 38 MB per layer) take about 1.17 ms, about 46% of peak: each kernel takes 8–10 µs against a 5 µs byte floor. **These GEMMs are too small to fill HBM; per-kernel latency dominates.** There are 820 kernels per decode step (counted per CUDA-graph launch; an earlier draft said about 835 by dividing the whole trace, prefill included), so the average kernel slot is about 5.5 µs at 8k. At batch 1, the number of kernels per step matters as much as bytes.

**3. Attention's fixed cost, split.** At 8k: FlashAttention-3 forward 0.61 ms (48 × 12.8 µs, about 40% of the KV byte floor's bandwidth), the split-KV combine kernel 0.15 ms, mRoPE 0.125 ms, QK-norm 0.067 ms, and the KV store 0.064 ms (counted as "other"). About 0.34 ms per step is small elementwise kernels around attention.

**4. Correction: cold-start variance is weight loading, not the JIT cache.** Server-log phases (container clock):

| Phase | Container 1 | Container 2 |
| --- | ---: | ---: |
| Stage env → weight load begin (process spawn, imports, audio-encoder graphs) | 38.8 s | 38.9 s |
| **Weight load (56.90 GB from the `omni-hf-cache` Volume)** | **176.9 s (0.32 GB/s)** | **47.3 s (1.20 GB/s)** |
| KV pool | 0.9 s | 0.8 s |
| Prefill CUDA graph capture (breakable backend, about 40 token-count buckets) | 33.6 s | 30.0 s |
| Decode CUDA graph capture (12 batch sizes) | 4.6 s | 4.3 s |
| Server warmup request | 17.8 s | 13.6 s |
| Server ready (from launch) | 282.3 s | 146.2 s |

Everything except weight loading is within 5 s between containers. Why the same Volume read ran 3.7× slower in one container is unknown (worker-local caching is a hypothesis, not tested). The earlier sentence crediting the JIT cache is corrected above.

**5. About 14 GB of GPU memory is never used, and KV memory limits capacity.** Peak sampled usage over both runs was 67,105 of 81,559 MiB, so **14.1 GiB was never touched**, including during 8k prefills. The thinker's `mem_fraction_static=0.784` is SGLang's automatic value minus Qwen3-Omni's `encoder_mem_reserve=0.05`. With it, the KV pool holds 25,161 tokens (2.3 GB), only **two** concurrent 9,000-token requests. Setting `--thinker.engine.mem_fraction_static` explicitly is allowed (`stages.py` pins it when set). If about 10 of the 14 GB went to KV, the pool would hold about 130,000 tokens, about 14 concurrent 8k/1k requests. That is untested: peak activation memory at higher concurrency has not been measured.

**6. Step-time tails are tight.** Client chunk gaps for L: p50 4.44, p90 4.68–4.72, p99 4.96–5.07, max 6.2–7.1 ms; one gap above 6 ms per 1,000 tokens.

**7. Smaller issues.** The roofline uses the 3.35 TB/s datasheet peak; the achievable copy bandwidth on this GPU was not measured, so every "% of peak" here is a lower bound on efficiency. The torch profiler logged "Failed to export trace: Trace is already saved" twice per container, but both traces were complete (1,000 decode steps each). py-spy cannot run on Modal (ptrace denied), so it is removed from later runs. SGLang-Omni has EAGLE3 speculative-decoding adapters at this pin, but none is wired to Qwen3-Omni and there is no public draft model; it is noted as the usual fix for latency-bound batch-1 decode, not planned.

**Revised predictions (agent, not the user's):**
- **FP8 at batch 1: 3.4–3.9 ms/token, not 2.8–2.9.** The expert GEMMs (about 70% of peak) should gain close to 2× on bytes, about 0.7–0.8 ms. The per-layer projections are latency-bound and gain little, about 0.2–0.3 ms. FP8 adds per-token activation-quantization kernels, plus 0.1–0.4 ms.
- **The tuned fused-MoE config is worth at most about 0.25 ms (5%)** if the expert GEMMs move from about 70% to about 85% of peak. It is still a cheap test and a possible upstream config PR, but it is not the main lever.
- **The biggest lever for a customer is capacity, not batch-1 latency.** Concurrency is capped at 2 by the KV pool; the memory fraction and FP8 weights and KV each raise it.

## Next

See the project README's "Revised plan after experiment 1". In short:
1. Experiment 2a/2b as planned, with the revised prediction and a non-copy quality probe; add a measured HBM copy bandwidth in the same container.
2. Experiment 4 gains step 4a: memory fraction and a closed-loop concurrency scan before the open-loop SLO sweep.
3. Experiment 3 targets weight loading and prefill CUDA-graph capture, which together are 77–210 s of the 146–282 s startup.
4. Optional: the fused-MoE tuning test (upper bound about 5%).
