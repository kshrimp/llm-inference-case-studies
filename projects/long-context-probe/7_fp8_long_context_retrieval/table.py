"""Needle-retrieval accuracy by length x depth, and paired comparisons between runs.

Reads results/long-context-probe/fp8-kv-needle/*.json written by `app.py --stage needle`
in experiment 3 (experiment 7 and the 7b ablations). Scoring: the 7-digit code appears in the output.
"""

import argparse
import collections
import json
from pathlib import Path


def key(record):
    return (record["target_length"], record["depth"], record["trial"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/long-context-probe/fp8-kv-needle"))
    args = parser.parse_args()
    runs = [json.loads(p.read_text()) for p in sorted(args.root.glob("*.json"))]
    for r in runs:
        name = r.get("ablation", r["variant"])
        cells = collections.defaultdict(lambda: [0, 0])
        for rec in r["records"]:
            cells[(rec["target_length"], rec["depth"])][0] += rec["correct"]
            cells[(rec["target_length"], rec["depth"])][1] += 1
        failures = [f"{rec['target_length']}@{rec['depth']}#{rec['trial']}" for rec in r["records"] if not rec["correct"]]
        print(f"{r['run_id']}\t{name}\tcache={r.get('cache_type', 'radix')}\tpool={r.get('kv_pool_capacity_tokens_excluding_dummy', 'n/a')}"
              f"\t{r['correct']}/{r['total']}\ttoken0_outputs={sum(x['token0_count'] > 0 for x in r['records'])}")
        print("   by cell:", " ".join(f"{L}@{d}:{c}/{n}" for (L, d), (c, n) in sorted(cells.items())))
        print("   failures:", failures or "none")
    reference = next((r for r in runs if r.get("ablation", r["variant"]) == "bf16"), None)
    if reference:
        ref = {key(x): x for x in reference["records"]}
        print(f"\npaired vs first BF16 run {reference['run_id']} (same prompts):")
        for r in runs:
            if r is reference:
                continue
            same = right_ref_only = right_this_only = 0
            for rec in r["records"]:
                base = ref.get(key(rec))
                if base is None:
                    continue
                same += base["text"] == rec["text"]
                right_ref_only += base["correct"] and not rec["correct"]
                right_this_only += rec["correct"] and not base["correct"]
            print(f"   {r.get('ablation', r['variant'])}\t{r['run_id']}\tidentical_text={same}/{len(r['records'])}"
                  f"\tref_only_right={right_ref_only}\tthis_only_right={right_this_only}")
