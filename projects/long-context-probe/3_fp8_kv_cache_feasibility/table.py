"""Summarize BF16/FP8 runs without merging repeats, and compare greedy quality fixtures."""

import argparse
import json
from pathlib import Path


def rows(root: Path):
    for path in sorted(root.glob("*/run.json")):
        yield path, json.loads(path.read_text())


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("results/long-context-probe/fp8-kv-cache-feasibility"))
    args = parser.parse_args()
    runs = list(rows(args.root))
    print("stage\tvariant\tinput\trepeat\tstatus\tactual/requested\telapsed_s\tkv_dtype"
          "\tpool_GiB\tpool_tokens\trequest_KV_GiB\toutput_sha\ttrace\trun_id")
    for path, data in runs:
        actual = data.get("actual_output_tokens")
        counts = f"{actual}/{data.get('requested_output_tokens')}" if actual is not None else "unknown"
        gib = lambda key: (f"{data[key] / 2**30:.3f}" if data.get(key) is not None else "null")
        print(f"{data.get('stage')}\t{data.get('variant')}\t{data.get('input_tokens', data.get('input_length_requested'))}"
              f"\t{data.get('repeat')}\t{data.get('status')}\t{counts}"
              f"\t{data.get('elapsed_seconds', 'null')}\t{data.get('kv_cache_dtype', 'unknown')}"
              f"\t{gib('kv_pool_allocated_bytes')}\t{data.get('kv_pool_capacity_tokens_excluding_dummy', 'null')}"
              f"\t{gib('logical_request_kv_bytes_estimate')}\t{data.get('output_sha256', 'unknown')[:12]}"
              f"\t{data.get('trace_file') or '-'}\t{data.get('run_id', path.parent.name)}")

    # Greedy quality: first token index where FP8 differs from the BF16 reference.
    reference = next((d for _, d in runs if d.get("variant") == "bf16" and d.get("quality_fixtures")), None)
    if reference is None:
        print("\nNo BF16 quality reference yet.")
    else:
        print(f"\nQuality vs BF16 run {reference['run_id']} (first divergent token index; none = identical)")
        for _, data in runs:
            if data.get("quality_fixtures") and data is not reference:
                indexes = [first_divergence(r["token_ids"], q["token_ids"])
                           for r, q in zip(reference["quality_fixtures"], data["quality_fixtures"])]
                print(f"{data['variant']}\t{data['run_id']}\t{indexes}")
