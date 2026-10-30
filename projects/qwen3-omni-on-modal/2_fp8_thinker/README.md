# Experiment 2: FP8 thinker at 8k / 1k

**Question.** Does FP8 make Qwen3-Omni's 8k/1k request faster on one H100 without losing quality, and how close does each condition get to its own bandwidth floor?

## Conditions

| Condition | Weights | KV cache | Floor (roofline) | Runs if |
| --- | --- | --- | --- | --- |
| A | BF16 (`Qwen/…-Instruct`) | BF16 | 2.07 ms/token | always |
| B | FP8 (`marksverdhei/Qwen3-Omni-30B-A3B-FP8`, community conversion; upstream FP8 config) | BF16 | 1.16 ms/token | the 2a smoke passes for B |
| C | FP8 | FP8 e4m3 | 1.03 ms/token | the CPU check shows `kv_cache_dtype=fp8_e4m3` in the resolved thinker args **and** the 2a smoke passes for C |

**Held equal across A, B, C:** thinker memory fraction (set explicitly, since upstream's FP8 config uses 0.75 and the BF16 default uses none), radix cache off, and server flags other than the model path and KV dtype. Each condition's KV token capacity is recorded from the server log.

## Step 2a: smoke and gate (one container)

As run (`app.py --stage fp8-smoke`, one container, servers started in the order B, C, A; A is the BF16 reference, because experiment 1 kept only output hashes): every server is text-only with `--thinker.engine.mem_fraction_static 0.784` (experiment 1's BF16 value) instead of upstream's colocated FP8 YAML, whose thinker fraction applies to the colocated speech layout. Per condition: the 8k prompt for 100 tokens and 3 free-generation prompts for 128 tokens via `/generate` (token ids and logprobs), plus one streamed 8k/1k request. The GPU probe measures the CUDA cast and device copy bandwidth first. **Correctness gate per condition:** no NaN or inf in logits (one probe request with logprobs), non-empty output, and at least 50% exact match with the stored BF16 reference over the first 100 tokens of the 8k/1k request. A condition that fails the gate is dropped and recorded as a finding. For C, the server log must also show an FP8 KV pool (the CPU check only proved the config accepts the override). The CUDA BF16 → e4m3 cast is probed once here (516 must map to 448).

## Step 2b: comparison (one container per condition)

> **Superseded by the redesign in [FINDINGS.md](FINDINGS.md#go--no-go-for-2b) (2026-10-28).** 2a showed that FP8 is 46–50% slower at batch 1. 2b now traces B and tests the dense-GEMM and MoE backend choices, in one container plus one repeat. The original design is kept below for the record.

Each container: 2 warmups, then 5 untraced 8k/1k requests, 1 traced request, and the quality set. A is run first and its outputs are the BF16 reference.

- **Speed:** prefill time, median decode ms/token, and the experiment 1 step anatomy for the traced request.
- **Quality set (held out):** 60 long-context retrieval items at 2k, 4k, and 8k context (20 each; the design of long-context experiment 7, rebuilt for this tokenizer, with the answer key generated before any run) plus exact match of the first 1,000 greedy tokens against A on the 5 measured requests.
- **Pass:** FP8 retrieval accuracy within the 95% Wilson interval of A's accuracy, and decode within 1.5× of its own floor. With 60 items the interval is wide (about ±12 points at 80%), so the result can show a large loss but not prove a small one. The report states this.

Loaded speed (8 and 32 concurrent requests) moves to experiment 4, which uses frozen pass criteria for that.

**Agent expectation (not the user's), revised after experiment 1:** B decodes at **3.4–3.9 ms/token** (BF16: 4.45). The expert GEMMs run at about 70% of peak bandwidth and should gain close to 2× on bytes (about −0.7 to −0.8 ms). The per-layer projection GEMMs are latency-bound at about 46% of peak and gain little (about −0.2 to −0.3 ms). FP8 adds per-token activation-quantization kernels (+0.1 to +0.4 ms). C adds little over B at batch 1: the KV byte floor at 8k is 0.25 ms, and most attention time is fixed cost. **FP8's larger value is probably capacity** (weights 57 → about 37 GB frees about 20 GB for KV), which experiment 4 measures.

## Changes after the experiment 1 review (2026-10-28)

- **Quality probe:** the copy-task outputs are a weak quality check, because copying is easy and a quantized model can copy perfectly while reasoning worse. Keep exact match on the 5 measured requests only as a sanity check. The pass decision rests on the retrieval set plus 20 free-generation prompts (greedy, 256 tokens), scored by exact match against A and by first divergence position.
- **Bandwidth calibration:** each 2b container first measures achievable HBM bandwidth with a large device-to-device `torch` copy (about 10 s). Every "% of peak" is then reported against both the datasheet 3.35 TB/s and the measured value.
- **Same analysis as experiment 1:** `analyze_trace.py` on the traced request, both overlap rules (`kernel_breakdown.py`), and new FP8 kernels (activation quant, block-scale GEMMs) categorized before the run. Unknown kernel names go to "other" and are listed.
- **No py-spy** (Modal denies ptrace).
