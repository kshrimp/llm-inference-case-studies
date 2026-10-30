"""Decode-step anatomy from an experiment 1 torch trace (trace clock only).

Kernel records overlap on the single stream (summed durations exceed the
busy time), so time is attributed exclusively: each busy instant goes to the
earliest-started kernel still running, and per-kernel durations are never
summed. Decode steps are the CUDA-graph launches with the most common kernel
count (>= 48); for BF16 these hold the 96 fused-MoE calls of one forward.

Usage: python analyze_trace.py <trace.json.gz> [<trace.json.gz> ...]
"""
import collections
import gzip
import json
import statistics as st
import sys

CATEGORIES = [  # first match wins
    # FP8 activation quantization (per-token-group, before every FP8 GEMM)
    ("fp8_quant", ("per_token_group_quant", "per_token_quant", "scaled_fp8_quant")),
    # MoE: Triton fused path, plus the CUTLASS grouped-GEMM path and its setup/permute
    # kernels (FillFunctor and bfloat16_copy appear once per layer only with CUTLASS MoE)
    ("moe", ("fused_moe", "moe_sum_reduce", "act_and_mul", "router", "moe_align", "topk",
             "count_and_sort", "moe_", "GroupProblemShape", "group_gemm_starts",
             "compute_arg_sorts", "compute_problem_sizes", "compute_expert_offsets",
             "shuffle_rows_with_scales", "shuffle_mul_sum", "FillFunctor<int>",
             "bfloat16_copy_kernel")),
    ("attention", ("FlashAttn", "flash::", "mrope", "qknorm", "rotary")),
    ("dense_gemm", ("nvjet", "cublas", "w8a8_block_fp8_matmul", "deep_gemm", "gemm",
                    "splitKreduce", "cutlass_80", "sm90_xmma")),
    ("norm", ("rmsnorm", "RMSNorm", "norm")),
    ("sampling", ("sampl", "argmax", "softmax", "top_k", "topp", "logits")),
]
MOE_PER_STEP = 96


def category(name: str) -> str:
    for cat, keys in CATEGORIES:
        if any(k in name for k in keys):
            return cat
    return "other"


def exclusive(kernels: list[dict], lo: float, hi: float) -> dict:
    """Busy time per category in [lo, hi), earliest-started kernel owns overlaps."""
    out: collections.Counter = collections.Counter()
    cursor = lo
    for k in sorted(kernels, key=lambda e: e["ts"]):
        s, e = max(k["ts"], cursor), min(k["ts"] + k["dur"], hi)
        if e > s:
            out[category(k["name"])] += e - s
            cursor = e
    return out


def analyze(path: str) -> dict:
    ev = json.load(gzip.open(path))["traceEvents"]
    launches = {e["args"]["correlation"]: e for e in ev
                if e.get("cat") == "cuda_runtime" and e["name"] == "cudaGraphLaunch"}
    by_corr = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") == "kernel":
            by_corr[e["args"].get("correlation")].append(e)
    # A decode step is a CUDA-graph launch with the most common kernel count among
    # launches of at least 48 kernels (one per layer); this holds for the BF16 fused-MoE
    # path (96 fused_moe_kernel calls) and for FP8 paths that use other MoE kernels.
    counts = collections.Counter(len(by_corr.get(c, [])) for c in launches)
    big = {n: k for n, k in counts.items() if n >= 48}
    step_kernels = max(big, key=big.get) if big else None
    steps = []
    for corr, launch in sorted(launches.items(), key=lambda kv: kv[1]["ts"]):
        ks = by_corr.get(corr, [])
        if ks and len(ks) == step_kernels:
            steps.append({"launch_ts": launch["ts"], "kernels": ks,
                          "start": min(k["ts"] for k in ks),
                          "end": max(k["ts"] + k["dur"] for k in ks)})
    rows = []
    all_kernels = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    for a, b in zip(steps, steps[1:]):
        period = b["start"] - a["start"]
        inside = [k for k in all_kernels if a["start"] <= k["ts"] < b["start"]]
        cats = exclusive(inside, a["start"], b["start"])
        rows.append({"period_us": period, "busy_us": sum(cats.values()),
                     "graph_us": a["end"] - a["start"], "cats": cats,
                     "launch_lead_us": a["start"] - a["launch_ts"]})
    return {"path": path, "decode_steps": len(steps), "kernels_per_step": step_kernels, "rows": rows}


def report(res: dict) -> None:
    rows = res["rows"]
    print(f"\n## {res['path']}\ndecode graph steps found: {res['decode_steps']} "
          f"({res['kernels_per_step']} kernels each); "
          f"step periods measured: {len(rows)}")
    med = lambda xs: st.median(xs) / 1000  # noqa: E731  (us -> ms)
    period = med([r["period_us"] for r in rows])
    busy = med([r["busy_us"] for r in rows])
    print(f"median step period {period:.3f} ms; GPU busy {busy:.3f} ms ({busy / period:.1%}); "
          f"idle {period - busy:.3f} ms; graph span {med([r['graph_us'] for r in rows]):.3f} ms; "
          f"launch-to-first-kernel {med([r['launch_lead_us'] for r in rows]):.3f} ms")
    cats = sorted({c for r in rows for c in r["cats"]})
    print("| Category | Median ms/step | Share of step |")
    print("| --- | ---: | ---: |")
    for c in cats:
        v = med([r["cats"].get(c, 0) for r in rows])
        print(f"| {c} | {v:.3f} | {v / period:.1%} |")
    print(f"| idle (no kernel) | {med([r['period_us'] - r['busy_us'] for r in rows]):.3f} | "
          f"{1 - busy / period:.1%} |")
    for lo, hi in ((0, 100), (100, 500), (500, len(rows))):
        part = rows[lo:hi]
        if part:
            print(f"steps {lo + 1}-{hi}: period {med([r['period_us'] for r in part]):.3f} ms, "
                  f"attention {med([r['cats'].get('attention', 0) for r in part]):.3f} ms")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(analyze(p))
