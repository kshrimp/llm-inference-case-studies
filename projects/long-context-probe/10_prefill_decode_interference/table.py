"""Per-config decode stall and latency for short requests next to long prefills, listed per repeat.

Reads results/long-context-probe/minisgl-prefill-decode-interference/*.json (`--stage interference`).
"""

import argparse
import json
from pathlib import Path

COLUMNS = ("short_gap_ms_max", "short_gap_ms_p99", "short_gap_ms_median", "short_first_decode_s_median",
           "short_completion_s_median", "short_completion_s_max", "elapsed_seconds", "prefill_steps", "steps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/long-context-probe/minisgl-prefill-decode-interference"))
    args = parser.parse_args()
    print("run_id\tconfig\t" + "\t".join(COLUMNS) + "\tlong_first_token_s")
    for path in sorted(args.root.glob("*.json")):
        run = json.loads(path.read_text())
        for c in run["configs"]:
            if "error" in c:
                print(f"{run['run_id']}\t{c['name']}\tERROR {c['error']}")
                continue
            values = "\t".join(f"{c[k]:.3f}" if isinstance(c[k], float) else str(c[k]) for k in COLUMNS)
            print(f"{run['run_id']}\t{c['name']}\t{values}\t{[round(x, 3) for x in c['long_first_token_s']]}")
