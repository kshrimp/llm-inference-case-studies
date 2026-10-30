# Experiment 6: FP8 KV break-even map (batch × context)

**Question.** When is FP8 KV worth using? Experiment 3 found −13.7% whole-call time for one 8,000-token request. Experiment 5 found +47.6% throughput at identical batching (about 69 concurrent requests). This grid maps where the gain grows, shrinks, or turns into a loss.

## Why we ran this

Experiments 3 and 5 gave two very different FP8 gains, −13.7% time for one request and +47.6% throughput at about 69 requests, without explaining the gap. A customer's question is not "is FP8 faster?" but "is it faster for *my* traffic?" The working hypothesis was that the gain depends on how many KV bytes a decode step reads, which is batch × context, and that at small loads the extra clamp and cast kernels could make FP8 slower. A grid that keeps capacity out of the picture tests that hypothesis and gives a break-even point a customer can compare with their own load.

## What we did

Ran the 16-cell grid below in four containers (BF16, FP8, FP8, BF16), with every cell sized to fit both pools so that each runs as one decode batch. Then [`table.py`](table.py) fit decode-step time against BF16 KV bytes in flight for each variant and solved for the break-even point and the saving ceiling.

**Design.** The runner is `--stage sweep` in [experiment 3's app.py](../3_fp8_kv_cache_feasibility/app.py). The images and pins are the same: BF16 is unmodified `mini-sglang@9a91cfaf`, and FP8 adds the saturating v2 patch (`1140df00…`).

- **Grid:** batch {1, 4, 16, 32} × context {256, 1,024, 2,048, 4,096} random tokens, with 256 output tokens per request, greedy decoding, and `ignore_eos`. That makes 16 cells, run after a discarded warmup cell.
- **Fits both pools:** the largest cell needs 32 × (4,096 + 256 + 255 page slack) ≈ 147,000 tokens, under the BF16 pool's 172,544. So every request in a cell runs in **one decode batch** in both variants, and pool capacity cannot confound the comparison. The runner records the decode batch max to confirm this.
- **Engine settings:** `max_seq_len_override=4608`, `max_extend_tokens=16384`, `cuda_graph_max_bs=32`, `page_size=256`.
- **Per cell:** wall time, output tok/s, time to the first decode batch (prefill and admission), median decode-step time over the middle half of decode (from decode-schedule timestamps; mini-SGLang's overlap scheduler makes this a steady-state rate, not a per-kernel time), decode batch max, and end KV occupancy in tokens and bytes.
- **Order:** BF16, FP8, FP8, BF16, one container each.

**Expected signal (agent-written; the user waived predictions on 2026-10-24).** FP8's step-time saving should grow with batch × context, because that product sets the KV bytes each decode step reads. At batch 1 × context 256, KV reads are tiny next to the 1.2 GB of weights per step, so FP8 should be neutral or slightly slower, because the extra clamp and cast kernels (about 0.18 ms/step in experiment 3) remain. A loss in any cell would be the most useful finding.

## How to run

Run from the repository root. `bash projects/long-context-probe/6_fp8_break_even/run.sh` runs everything below in order.

```bash
cd /path/to/llm-inference-case-studies
A=projects/long-context-probe/3_fp8_kv_cache_feasibility/app.py
uv run modal run $A::main --stage sweep --variant bf16 --repeat 1
uv run modal run $A::main --stage sweep --variant fp8_sat --repeat 1
uv run modal run $A::main --stage sweep --variant fp8_sat --repeat 2
uv run modal run $A::main --stage sweep --variant bf16 --repeat 2
uv run python projects/long-context-probe/6_fp8_break_even/table.py
uv run modal app list
```

Cost: 4 containers × $0.34 worst case. Results go to `results/long-context-probe/fp8-kv-break-even/`.

## Files

- [`run.sh`](run.sh): the launch order (BF16, FP8, FP8, BF16).
- [`table.py`](table.py): per-cell table, linear step-time fits, break-even, and ceiling, from `results/long-context-probe/fp8-kv-break-even/`.
- Runner: `--stage sweep` in [experiment 3's app.py](../3_fp8_kv_cache_feasibility/app.py) (`sweep_case`).
- Results and analysis: [FINDINGS.md](FINDINGS.md).
