# Experiment 1: long-context anatomy (8k in / 1k out, BF16)

**Question.** On SGLang-Omni at the current pin, where does the time of one 8,000-input / 1,000-output thinker request go? How does decode-step time depend on context length, separately from step position?

## Design

- Text-only server (`--text-only`, default memory settings, no colocated profile), BF16, radix cache off, greedy decoding. As run: copy-task prompts longer than the 1,000-token cap stand in for `ignore_eos`, which the chat API lacks (see Files).
- **Prompts:** a fixed 8,000-token prompt (**L**) and a fixed 1,150-token prompt (**S**, the control; 1,150 rather than 1,000 so the copy task outlasts the cap) from a deterministic text source, checked by token count with the pinned tokenizer and stored with their SHA-256.
- **Per container:** 2 unmeasured warmups, then L, S, L, S, L, S, L, S, L, S untraced (5 each, interleaved so drift affects both), then 1 traced L and 1 traced S (torch profiler, CUDA + CPU activities), then 1 L under `py-spy record --native` on the thinker process.
- **Two containers**, the second an independent repeat. A conclusion needs both containers to agree within the stop condition.
- **Recorded per request** (client clock): TTFT, every streamed-chunk arrival time, completion time. Per traced request (trace clock): every decode step's start and end, from the step-level marker in the trace (identify the marker in the CPU check; if none exists, use the first kernel of each forward pass).

## Analysis

1. **Prefill vs decode.** Prefill time = first-token time minus dispatch; decode = the remaining 999 steps. Reported from untraced runs.
2. **Context effect.** At each decode step index *k*, L has context 8,000 + *k* and S has 1,000 + *k*. The per-step difference L − S at equal *k* isolates the 7,000-token context difference from step position (batch state, allocator state, and graph choice are the same at equal *k*). Expected from bytes alone: 7,000 × 96 KiB ≈ 0.69 GB more KV read per step, about 0.2 ms at peak bandwidth. A larger difference means attention is not bandwidth-bound.
3. **Step anatomy (traced L, median step).** Assign trace time to: MoE GEMM kernels, attention kernels, other GPU kernels, CPU-side ops recorded in the trace, and GPU idle with no recorded CPU op. Whatever no category covers is **unattributed** and is reported as such. The 95% target is met only if unattributed is ≤ 5%.
4. **py-spy is for identification, not accounting.** Its samples name which Python and native functions dominate the idle gaps. They do not assign milliseconds.
5. **Bound check.** Compare the median step with the 2.07 ms floor and say which categories hold the excess. Check whether async decode (upstream PR #908) and CUDA graphs are active in the resolved config.

**Agent expectation (not the user's; predictions waived ):** decode is 3.5 to 5 ms/token. The fused MoE kernels at M = 1 hold most of the excess over the floor, and the L − S step difference is larger than 0.2 ms.

**Validity gates.** Every request returns exactly 1,000 tokens with no errors. Prompt token counts match the stored prompts. If the traced median step differs from the untraced median step by more than 15%, report the profiler overhead and use untraced numbers for all totals.

## Files

`../app.py --stage anatomy` (one container), `run.sh` (exact commands), `table.py` (client numbers), `analyze_trace.py` (trace anatomy), `FINDINGS.md`. The runner sizes prompts with the server's token count and uses copy tasks so outputs stop on length (no `ignore_eos` in the chat API); the short control is 1,150 tokens, not 1,000, for that reason.
