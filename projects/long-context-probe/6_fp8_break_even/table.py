"""Break-even map: per-cell BF16 vs FP8 throughput and step time, plus a linear KV-bandwidth fit.

Reads results/long-context-probe/fp8-kv-break-even/*.json written by `app.py --stage sweep`
in experiment 3. Step time is regressed on KV bytes in flight at mid-decode:
batch x (context + output/2) tokens x bytes per token.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

BYTES_PER_TOKEN = {"bf16": 114688, "fp8_sat": 57344}  # 2 x 28 layers x 8 heads x 128 dims x dtype bytes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/long-context-probe/fp8-kv-break-even"))
    args = parser.parse_args()
    runs = [json.loads(p.read_text()) for p in sorted(args.root.glob("*.json"))]
    print("runs:", [(r["run_id"], r["variant"]) for r in runs])
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    points = collections.defaultdict(list)
    for r in runs:
        half_output = r["output_tokens_per_request"] // 2
        for c in r["cells"]:
            if not c["valid_count"] or c["decode_batch_max"] != c["batch"]:
                print("WARNING: cell not a single full batch or short output:", r["run_id"], c["batch"], c["context"])
            cells[(c["batch"], c["context"])][r["variant"]].append(c)
            gb = c["batch"] * (c["context"] + half_output) * BYTES_PER_TOKEN[r["variant"]] / 1e9
            points[r["variant"]].append((gb, c["decode_step_ms_median_middle_half"]))
    print("\ncontext\tbatch\tBF16_KV_GB_end\tBF16_tok_s(r1,r2)\tFP8_tok_s(r1,r2)\ttput_change\tstep_ms_BF16->FP8\tstep_change")
    for (b, ctx) in sorted(cells, key=lambda k: (k[1], k[0])):
        bf, f8 = cells[(b, ctx)]["bf16"], cells[(b, ctx)]["fp8_sat"]
        tb, tf = [c["output_tok_s"] for c in bf], [c["output_tok_s"] for c in f8]
        sb = statistics.mean(c["decode_step_ms_median_middle_half"] for c in bf)
        sf = statistics.mean(c["decode_step_ms_median_middle_half"] for c in f8)
        print(f"{ctx}\t{b}\t{bf[0]['kv_occupancy_bytes_end'] / 1e9:.2f}\t{tb[0]:.1f},{tb[1]:.1f}\t{tf[0]:.1f},{tf[1]:.1f}\t"
              f"{100 * (statistics.mean(tf) / statistics.mean(tb) - 1):+.1f}%\t{sb:.2f}->{sf:.2f}\t{100 * (sf / sb - 1):+.1f}%")
    fits = {}
    for variant, xy in points.items():
        x, y = [a for a, _ in xy], [b for _, b in xy]
        slope, intercept = statistics.linear_regression(x, y)
        fits[variant] = (intercept, slope)
        print(f"\n{variant}: step_ms = {intercept:.2f} + {slope:.2f} * KV_GB  (R^2 {statistics.correlation(x, y) ** 2:.3f}); "
              f"effective KV read {1000 / slope:.0f} GB/s", end="")
    if {"bf16", "fp8_sat"} <= fits.keys():
        (ib, sb), (if8, sf8) = fits["bf16"], fits["fp8_sat"]
        breakeven_gb = (if8 - ib) / (sb - sf8 / 2)
        print(f"\n\nbreak-even: {breakeven_gb:.3f} GB of BF16 KV per step = {breakeven_gb * 1e9 / BYTES_PER_TOKEN['bf16']:.0f} tokens in flight; "
              f"step-time ceiling {100 * (1 - sf8 / (2 * sb)):.0f}%")
