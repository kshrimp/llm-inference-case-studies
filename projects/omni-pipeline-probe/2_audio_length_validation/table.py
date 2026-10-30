"""Experiment 2 table: 10 / 30 / 90 s audio, every repeat and run (local).

Usage: uv run python projects/omni-pipeline-probe/2_audio_length_validation/table.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import load_rows, median, print_table, run_dirs  # noqa: E402

COLUMNS = [
    ("run", "run_short"), ("step", "step"), ("valid", "valid"), ("audio s", "duration_s"),
    ("prompt tok", "prompt_tokens"), ("out tok", "completion_tokens"), ("client TTFT", "ttft_ms"),
    ("completion", "completion_ms"), ("preprocess", "preprocess_ms"), ("audio enc", "audio_encoder_ms"),
    ("admit→thinker queue", "admission_to_thinker_queue_ms"), ("first forward", "thinker_first_forward_ms"),
    ("first output", "thinker_first_output_ms"), ("ms/token", "decode_ms_per_token"),
    ("chunk gap", "thinker_chunk_gap_median_ms"),
]
METRICS = ["ttft_ms", "completion_ms", "preprocess_ms", "audio_encoder_ms",
           "admission_to_thinker_queue_ms", "thinker_first_forward_ms", "decode_ms_per_token", "thinker_chunk_gap_median_ms"]


def main() -> None:
    rows = []
    for run_dir in run_dirs():
        info, run_rows = load_rows(run_dir)
        print(f"run {info['run_id']}: all cached tokens zero: {info['all_cached_tokens_zero']}")
        rows += [dict(r, run_short=r["run_id"][:16]) for r in run_rows if r["experiment"] == "exp2"]
    if not rows:
        sys.exit("no measurement runs found")
    rows.sort(key=lambda r: (r["duration_s"], r["run_id"], r["step"]))
    print("\nAll times in ms; individual repeats. Event spans: server time_ns clock.\n")
    print_table(rows, COLUMNS)
    for r in rows:
        if not r["valid"]:
            print(f"INVALID {r['run_short']} {r['step']}: {r['invalid_reasons']}")
    print("\nMedian per duration over valid repeats, and the 10 s → 90 s change (descriptive):\n")
    by = {}
    for d in (10, 30, 90):
        valid = [r for r in rows if r["duration_s"] == d and r["valid"]]
        by[d] = {"audio_s": d, "n": len(valid), **{k: median([r[k] for r in valid]) for k in METRICS}}
    change = {"audio_s": "90−10", "n": None}
    for k in METRICS:
        a, b = by[10][k], by[90][k]
        change[k] = round(b - a, 3) if a is not None and b is not None else None
    print_table([by[10], by[30], by[90], change],
                [("audio s", "audio_s"), ("n", "n")] + [(k, k) for k in METRICS])


if __name__ == "__main__":
    main()
