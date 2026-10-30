"""Re-score saved arithmetic-eval records from their output text (last integer = answer).

Records saved on 2026-10-24 stored `accuracy` from a first-integer parse, which scored
answers like "85 * 32 = 2720" as 85; this script is the authoritative scorer.
"""

import argparse
import collections
import json
import re
from pathlib import Path


def last_int(text: str):
    found = re.findall(r"-?\d+", text.replace(",", ""))
    return int(found[-1]) if found else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/long-context-probe/fp8-kv-accuracy-eval"))
    args = parser.parse_args()
    runs = {}
    for path in sorted(args.root.glob("*.json")):
        data = json.loads(path.read_text())
        runs[data["variant"]] = data
    correct = {}
    for variant, data in runs.items():
        ok = [last_int(r["text"]) == r["answer"] for r in data["records"]]
        correct[variant] = ok
        by_op, total = collections.Counter(), collections.Counter()
        for record, hit in zip(data["records"], ok):
            op = record["question"].split()[3]
            total[op] += 1
            by_op[op] += hit
        print(f"{variant}\t{sum(ok)}/{len(ok)}\t{100 * sum(ok) / len(ok):.1f}%\t"
              + " ".join(f"{op}:{by_op[op]}/{total[op]}" for op in sorted(total))
              + f"\ttoken0_outputs:{sum(r['token0_count'] > 0 for r in data['records'])}\t{data['run_id']}")
    if "bf16" in correct:
        for variant in correct:
            if variant == "bf16":
                continue
            pairs = list(zip(correct["bf16"], correct[variant]))
            same_text = sum(a["text"] == b["text"] for a, b in zip(runs["bf16"]["records"], runs[variant]["records"]))
            print(f"vs bf16: {variant}\tboth_right:{sum(a and b for a, b in pairs)}\tbf16_only:{sum(a and not b for a, b in pairs)}"
                  f"\t{variant}_only:{sum(b and not a for a, b in pairs)}\tidentical_text:{same_text}/{len(pairs)}")
