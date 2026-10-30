"""Summarize completed phase-two runs without merging repeats or workloads."""

import argparse
import json
from pathlib import Path


def rows(root: Path):
    for path in sorted(root.glob("*/run.json")):
        data = json.loads(path.read_text())
        yield path, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("results/long-context-probe/context-length-validation"))
    args = parser.parse_args()
    print("measurement\tinput\trepeat\tstatus\tactual/requested\telapsed_s\toutput_sha\tcommit\trun_id")
    for path, data in rows(args.root):
        actual = data.get("actual_output_tokens")
        requested = data.get("requested_output_tokens")
        counts = f"{actual}/{requested}" if actual is not None else "unknown"
        elapsed = data.get("elapsed_seconds")
        print(f"{data.get('measurement', 'unknown')}\t{data.get('input_tokens', data.get('input_length_requested'))}"
              f"\t{data.get('repeat')}\t{data.get('status')}\t{counts}"
              f"\t{elapsed if elapsed is not None else 'null'}"
              f"\t{data.get('output_sha256', 'unknown')[:12]}"
              f"\t{data.get('engine_commit', 'unknown')[:12]}\t{data.get('run_id', path.parent.name)}")
