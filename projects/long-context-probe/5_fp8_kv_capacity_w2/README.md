# Experiment 5: FP8 KV capacity on the W2 offline throughput benchmark

**Question.** Does storing KV in FP8 (the experiment-3 v2 saturating patch) raise mini-SGLang's offline throughput on the sprint's W2 benchmark, and how much of any gain comes from the doubled KV capacity versus cheaper attention reads?

## Why we ran this

Experiment 3 showed that saturated FP8 is correct and cuts per-step attention time, but only for one request. That leaves FP8's other benefit, a 2× larger KV pool, untested, even though it is the usual reason to adopt FP8 KV in serving. The W2 offline benchmark is the sprint's agreed throughput baseline, so measuring FP8 there connects the tweak to a number already on record. A gain there could come from two causes, though: cheaper attention reads or more concurrent requests. A customer deciding whether to enable FP8 needs to know which one, because only the second depends on the workload being memory-bound. That is why the experiment has a capacity-matched control.

Experiment 3 measured one request, so the doubled pool did nothing there. In pinned mini-SGLang, `scheduler/prefill.py::_try_allocate_one` admits a request only if `input_len + output_len`, plus the tokens already reserved for running requests, fits in the free pool. The W2 benchmark asks for 276,793 input + output tokens across its 256 requests, more than the BF16 pool's 172,544. So BF16 cannot hold all requests at once, while the FP8 pool can hold nearly all of them.

## Setup

- **Runner:** `--stage bench` in [experiment 3's app.py](../3_fp8_kv_cache_feasibility/app.py). It uses the same images as experiment 3: BF16 is unmodified `mini-sglang@9a91cfaf`, and FP8 is that commit plus [`fp8_kv_cache_v2_saturate.patch`](../3_fp8_kv_cache_feasibility/fp8_kv_cache_v2_saturate.patch) (sha256 `1140df00…`). Both pin torch 2.9.1, flashinfer-python 0.7.2.post1, sgl-kernel 0.3.21, and transformers 4.57.3.
- **Workload:** reproduced in-process from upstream `benchmark/offline/bench.py` (MIT), with the same `seed(0)`, request generator (256 sequences, 100–1,024 input and 100–1,024 output tokens, temperature 0.6, `ignore_eos`), engine settings (`max_seq_len_override=4096`, `max_extend_tokens=16384`, `cuda_graph_max_bs=256`, `page_size=256`), warmup, and `total_tokens / time` formula. FP8 adds only `kv_cache_dtype="fp8_e4m3"`.
- **Study instrumentation:** a wrapper on the decode manager's `schedule_next_batch` records each decode batch's size. It observes and does not change scheduling.

## What we did

1. **Full-capacity comparison (2026-10-24 morning):** BF16, FP8, FP8, BF16, one container each, on the W2 workload. FP8 got its natural 2× pool (345,344 tokens).
2. **Capacity-matched control (2026-10-24 evening):** BF16, FP8 capped at BF16's 674 pages (`fp8_sat_capmatch`), FP8 capped, BF16. With the same pool, the scheduler makes the same batches, so any remaining gain comes from cheaper attention alone.
3. Rebuilt both tables with [`table.py`](table.py) and split the gain into its attention and capacity parts. Accuracy came from experiment 3's arithmetic check.

## How to run

Run from the repository root. [`run.sh`](run.sh) runs all eight containers in order and then the table:

```bash
cd /path/to/llm-inference-case-studies
bash projects/long-context-probe/5_fp8_kv_capacity_w2/run.sh
```

Or run them one at a time:

```bash
A=projects/long-context-probe/3_fp8_kv_cache_feasibility/app.py
# Full-capacity FP8 vs BF16
uv run modal run $A::main --stage bench --variant bf16 --repeat 1
uv run modal run $A::main --stage bench --variant fp8_sat --repeat 1
uv run modal run $A::main --stage bench --variant fp8_sat --repeat 2
uv run modal run $A::main --stage bench --variant bf16 --repeat 2
# Capacity-matched control (FP8 pool capped at BF16's 674 pages)
uv run modal run $A::main --stage bench --variant bf16 --repeat 3
uv run modal run $A::main --stage bench --variant fp8_sat_capmatch --repeat 1
uv run modal run $A::main --stage bench --variant fp8_sat_capmatch --repeat 2
uv run modal run $A::main --stage bench --variant bf16 --repeat 4
uv run python projects/long-context-probe/5_fp8_kv_capacity_w2/table.py
uv run modal app list
```

Raw records are in `results/long-context-probe/fp8-kv-capacity-w2/`. Each run is one container with a $0.34 worst-case reservation.


## Files

- [`run.sh`](run.sh): the exact launch order: BF16, FP8, FP8, BF16, then the capacity-matched control BF16, capmatch, capmatch, BF16.
- [`table.py`](table.py): rebuilds the run table and the gain split from `results/long-context-probe/fp8-kv-capacity-w2/`.
- Runner: `--stage bench` in [experiment 3's app.py](../3_fp8_kv_cache_feasibility/app.py), with `bench_case` and the `fp8_sat_capmatch` variant.
- Results and analysis: [FINDINGS.md](FINDINGS.md).
