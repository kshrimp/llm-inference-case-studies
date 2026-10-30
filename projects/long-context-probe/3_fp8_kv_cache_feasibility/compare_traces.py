"""Compare matched BF16 and FP8 128-output traces kernel by kernel and per decode step.

Reuses the experiment-1 trace reader and decode-step analyzer. Kernel sums can
overlap and steps are inferred from kernel order; this is diagnostic, not wall time.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_profile_baseline"))
from analyze_trace import trace_events  # noqa: E402
from decode_step_trend import summarize as decode_steps  # noqa: E402


def short_name(name: str) -> str:
    name = re.sub(r"^void ", "", name)
    return re.split(r"[<(]", name, maxsplit=1)[0][:80]


def kernel_totals(path: Path):
    totals = collections.defaultdict(lambda: [0, 0.0])
    for event in trace_events(path):
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            entry = totals[short_name(event.get("name", ""))]
            entry[0] += 1
            entry[1] += event.get("dur", 0) / 1000
    return totals


def compare(bf16: Path, fp8: Path, window: int, top: int):
    steps = {name: decode_steps(path, 28, window) for name, path in (("bf16", bf16), ("fp8", fp8))}
    totals = {"bf16": kernel_totals(bf16), "fp8": kernel_totals(fp8)}
    rows = []
    for kernel in set(totals["bf16"]) | set(totals["fp8"]):
        b_count, b_ms = totals["bf16"].get(kernel, (0, 0.0))
        f_count, f_ms = totals["fp8"].get(kernel, (0, 0.0))
        rows.append({"kernel": kernel, "bf16_count": b_count, "bf16_ms": round(b_ms, 3),
                     "fp8_count": f_count, "fp8_ms": round(f_ms, 3),
                     "delta_ms": round(f_ms - b_ms, 3)})
    rows.sort(key=lambda r: abs(r["delta_ms"]), reverse=True)
    per_step = {}
    for key in ("decode_attention_kernel_sum_per_step", "decode_attention_start_interval",
                "main_gemv_kernel_sum_per_step"):
        if key in steps["bf16"] and key in steps["fp8"]:
            b, f = steps["bf16"][key]["first_median_ms"], steps["fp8"][key]["first_median_ms"]
            per_step[key] = {"bf16_first_median_ms": b, "fp8_first_median_ms": f,
                             "change_percent": round(100 * (f / b - 1), 2)}
    return {
        "bf16_trace": str(bf16), "fp8_trace": fp8.as_posix(), "window_steps": window,
        "decode_steps_inferred": {k: v["decode_steps_inferred"] for k, v in steps.items()},
        "per_step_first_window": per_step,
        "kernel_totals_by_abs_delta": rows[:top],
        "limits": "Kernel sums overlap and include profiler overhead; one trace per variant.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bf16_trace", type=Path)
    parser.add_argument("fp8_trace", type=Path)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(compare(args.bf16_trace, args.fp8_trace, args.window, args.top), indent=2))
