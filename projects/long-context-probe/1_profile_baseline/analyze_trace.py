"""Stream a PyTorch Chrome trace and report descriptive event totals.

Totals within and across categories may overlap; they are not wall time.
"""

import argparse
import collections
import json
from pathlib import Path


def trace_events(path: Path):
    decoder = json.JSONDecoder()
    with path.open() as source:
        for line in source:
            if '"traceEvents": [' in line:
                break
        else:
            raise ValueError("traceEvents array missing")
        buffer = ""
        for chunk in iter(lambda: source.read(65536), ""):
            buffer += chunk
            while True:
                buffer = buffer.lstrip(" \r\n\t,")
                if not buffer or buffer[0] == "]":
                    break
                try:
                    event, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                yield event
                buffer = buffer[end:]
            if buffer.startswith("]"):
                return


def group(event):
    name = event.get("name", "")
    category = event.get("cat", "")
    if category == "kernel":
        if "BatchPrefillWithPagedKVCacheKernel" in name:
            return "GPU prefill attention"
        if "BatchDecodeWithPagedKVCacheKernel" in name:
            return "GPU decode attention"
        if "gemvx::kernel" in name:
            return "GPU cuBLAS GEMV"
        return "GPU other kernels"
    if category == "cuda_runtime" and name == "cudaEventSynchronize":
        return "CPU CUDA synchronization calls"
    return None


def summarize(path: Path):
    counts = collections.Counter()
    duration_us = collections.Counter()
    categories = collections.Counter()
    for event in trace_events(path):
        if event.get("ph") != "X":
            continue
        categories[event.get("cat", "unknown")] += 1
        label = group(event)
        if label:
            counts[label] += 1
            duration_us[label] += event.get("dur", 0)
    return {
        "trace": str(path),
        "category_event_counts": dict(categories),
        "groups": {label: {"events": counts[label], "sum_ms": round(duration_us[label] / 1000, 3)}
                   for label in sorted(counts)},
        "interpretation": "Event sums may overlap; do not treat them as wall time or causal speedups.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trace), indent=2))
