"""Summarize W2 throughput runs (BF16, FP8, capacity-matched FP8) and split the FP8 gain.

Reads results/long-context-probe/fp8-kv-capacity-w2/*.json written by
`app.py --stage bench` in experiment 3. Repeats are listed separately, never merged.
"""

import argparse
import json
import statistics
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/long-context-probe/fp8-kv-capacity-w2"))
    args = parser.parse_args()
    runs = [json.loads(p.read_text()) for p in sorted(args.root.glob("*.json"))]
    print("run_id\tvariant\tpool_tokens\tgenerated/requested\ttime_s\ttok_s\tdecode_steps\tbatch_max/mean")
    by_variant = {}
    for r in runs:
        print(f"{r['run_id']}\t{r['variant']}\t{r['kv_pool_capacity_tokens_excluding_dummy']}\t"
              f"{r['generated_tokens']}/{r['requested_tokens']}\t{r['elapsed_seconds']:.3f}\t{r['throughput_tok_s']:.2f}\t"
              f"{r['decode_steps']}\t{r['decode_batch_size_max']}/{r['decode_batch_size_mean']:.2f}")
        by_variant.setdefault(r["variant"], []).append(r)
    bf16 = by_variant.get("bf16", [])
    # Session pairing: r1/r2 bracket full FP8; r3/r4 bracket the capacity-matched control.
    pair_full = [r["throughput_tok_s"] for r in bf16 if r["repeat"] in (1, 2)]
    pair_cap = [r["throughput_tok_s"] for r in bf16 if r["repeat"] in (3, 4)]
    full = [r["throughput_tok_s"] for r in by_variant.get("fp8_sat", [])]
    cap = [r["throughput_tok_s"] for r in by_variant.get("fp8_sat_capmatch", [])]
    if pair_full and full:
        print(f"\nfull FP8 vs paired BF16: {statistics.mean(full):.2f} vs {statistics.mean(pair_full):.2f} "
              f"-> {100 * (statistics.mean(full) / statistics.mean(pair_full) - 1):+.1f}%")
    if pair_cap and cap:
        print(f"capacity-matched FP8 vs paired BF16: {statistics.mean(cap):.2f} vs {statistics.mean(pair_cap):.2f} "
              f"-> {100 * (statistics.mean(cap) / statistics.mean(pair_cap) - 1):+.1f}% (cheaper attention at identical batching)")
    if full and cap:
        print(f"full FP8 vs capacity-matched FP8: {100 * (statistics.mean(full) / statistics.mean(cap) - 1):+.1f}% (2x concurrency)")
    if bf16:
        t = [r["throughput_tok_s"] for r in bf16]
        print(f"all BF16 runs: {min(t):.2f}-{max(t):.2f} tok/s ({100 * (max(t) / min(t) - 1):.1f}% spread)")
