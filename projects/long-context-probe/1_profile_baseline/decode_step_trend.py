"""Summarize decode attention and GEMV work by step from a single-request trace.

Assumes the pinned Qwen3-0.6B has 28 layers and one decode-attention kernel
per layer per decode step. Rejects traces that do not match that pattern.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

from analyze_trace import trace_events


def describe(values, window):
    if len(values) < 2 * window:
        raise ValueError(f"need at least {2 * window} steps, got {len(values)}")
    return {
        "first_median_ms": round(statistics.median(values[:window]), 4),
        "last_median_ms": round(statistics.median(values[-window:]), 4),
        "change_percent": round(
            100 * (statistics.median(values[-window:]) / statistics.median(values[:window]) - 1), 2
        ),
    }


def summarize(path, layers, window):
    attention = []
    gemv_by_name = collections.defaultdict(list)
    for event in trace_events(path):
        if event.get("cat") != "kernel":
            continue
        name = event.get("name", "")
        if "BatchDecodeWithPagedKVCacheKernel" in name:
            attention.append((event["ts"], event["dur"]))
        elif "gemvx::kernel" in name:
            gemv_by_name[name].append((event["ts"], event["dur"]))
    attention.sort()
    if not attention or len(attention) % layers:
        raise ValueError("decode-attention event count is not a multiple of layer count")
    steps = len(attention) // layers
    attention_ms = [sum(d for _, d in attention[i:i + layers]) / 1000
                    for i in range(0, len(attention), layers)]
    starts = [attention[i][0] for i in range(0, len(attention), layers)]
    interval_ms = [(end - start) / 1000 for start, end in zip(starts, starts[1:])]
    result = {
        "trace": str(path), "decode_steps_inferred": steps,
        "attention_kernels_per_step": layers, "window_steps": window,
        "decode_attention_kernel_sum_per_step": describe(attention_ms, window),
        "decode_attention_start_interval": describe(interval_ms, window),
    }
    if gemv_by_name:
        main_gemv = max(gemv_by_name.values(), key=len)
        if len(main_gemv) % steps == 0:
            per_step = len(main_gemv) // steps
            main_gemv.sort()
            gemv_ms = [sum(d for _, d in main_gemv[i:i + per_step]) / 1000
                       for i in range(0, len(main_gemv), per_step)]
            result["main_gemv_kernel_sum_per_step"] = {
                "kernels_per_step": per_step, **describe(gemv_ms, window)
            }
    result["limits"] = (
        "Steps are inferred from kernel count and timestamp order, not explicit markers. "
        "Kernel sums can overlap; intervals include other GPU/CPU work. "
        "One trace is diagnostic, not a performance repeat or bandwidth proof."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trace, args.layers, args.window), indent=2))
