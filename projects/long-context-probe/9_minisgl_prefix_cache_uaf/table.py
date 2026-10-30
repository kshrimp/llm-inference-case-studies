"""Experiment 9 summary: regression-test outcomes, use-after-free detector events, and fixed-engine accuracy.

Needle prompt index == request uid: the needle stage submits all prompts in one generate call
on a fresh engine, so mini-SGLang assigns uids 0..N-1 in prompt order.
"""

import argparse
import collections
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-root", type=Path, default=Path("results/long-context-probe/minisgl-prefix-cache-uaf"))
    parser.add_argument("--needle-root", type=Path, default=Path("results/long-context-probe/fp8-kv-needle"))
    args = parser.parse_args()
    for path in sorted(args.unit_root.glob("*unittest*.json")):
        r = json.loads(path.read_text())
        summary = [line for line in r["stdout_tail"].splitlines() if "passed" in line or "failed" in line]
        print(f"unit test {r['variant']}\treturncode={r['returncode']}\t{summary[-1] if summary else 'no summary'}\t{r['run_id']}")
    needle = {}
    for path in sorted(args.needle_root.glob("*.json")):
        r = json.loads(path.read_text())
        needle.setdefault(r.get("ablation", r["variant"]), r)  # first run of each ablation
    label = lambda rec: f"{rec['target_length']}@{rec['depth']}#{rec['trial']}"
    base = needle.get("bf16")
    base_failures = {i for i, rec in enumerate(base["records"]) if not rec["correct"]} if base else set()
    if base:
        print(f"\nBF16 radix baseline {base['run_id']}: {base['correct']}/{base['total']}; failed:", sorted(label(base['records'][i]) for i in base_failures))
    detect = needle.get("bf16_uafdetect")
    if detect:
        events = detect.get("uaf_events") or []
        victims = collections.Counter(e["victim_uid"] for e in events)
        failed = {i for i, rec in enumerate(detect["records"]) if not rec["correct"]}
        print(f"\ndetector {detect['run_id']}: {detect['correct']}/{detect['total']}; {len(events)} live-page reallocation events, "
              f"{len(victims)} victim requests")
        print("   failed prompts that were victims:", sorted(label(detect['records'][i]) for i in failed & set(victims)))
        print("   failed prompts NOT victims:", sorted(label(detect['records'][i]) for i in failed - set(victims)))
        print("   victims that still answered correctly:", len(set(victims) - failed))
    fixed = needle.get("bf16_cachefix")
    if fixed:
        failed = {i for i, rec in enumerate(fixed["records"]) if not rec["correct"]}
        print(f"\nfixed engine {fixed['run_id']}: {fixed['correct']}/{fixed['total']}; failed:", sorted(label(fixed['records'][i]) for i in failed))
        if base:
            print("   baseline failures recovered:", len(base_failures - failed), "of", len(base_failures))
        naive = needle.get("bf16_naive")
        if naive:
            same = sum(a["text"] == b["text"] for a, b in zip(naive["records"], fixed["records"]))
            print(f"   identical text to the naive-cache run: {same}/{len(fixed['records'])}")
