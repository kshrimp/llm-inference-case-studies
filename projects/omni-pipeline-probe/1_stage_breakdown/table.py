"""Experiment 1 table: text control vs 30 s audio, every repeat and run (local).

Usage: uv run python projects/omni-pipeline-probe/1_stage_breakdown/table.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import load_rows, median, print_table, run_dirs  # noqa: E402

COLUMNS = [
    ("run", "run_short"), ("step", "step"), ("valid", "valid"), ("prompt tok", "prompt_tokens"),
    ("out tok", "completion_tokens"), ("client TTFT", "ttft_ms"), ("completion", "completion_ms"),
    ("dispatch→admit", "client_dispatch_to_admission_ms"), ("preprocess", "preprocess_ms"),
    ("audio enc", "audio_encoder_ms"), ("admit→thinker queue", "admission_to_thinker_queue_ms"),
    ("thinker queue", "thinker_queue_ms"), ("first forward", "thinker_first_forward_ms"),
    ("first output", "thinker_first_output_ms"), ("ms/token", "decode_ms_per_token"),
    ("chunk gap", "thinker_chunk_gap_median_ms"),
]
SUMMARY = ["ttft_ms", "completion_ms", "preprocess_ms", "audio_encoder_ms",
           "admission_to_thinker_queue_ms", "thinker_first_forward_ms", "thinker_first_output_ms",
           "decode_ms_per_token", "thinker_chunk_gap_median_ms"]


def main() -> None:
    rows = []
    for run_dir in run_dirs():
        info, run_rows = load_rows(run_dir)
        print(f"run {info['run_id']}: {info['gpu']}; ready {info['server_ready_s']} s; "
              f"text control {info['text_control'] and {k: info['text_control'][k] for k in ('words', 'prompt_tokens')}}; "
              f"all cached tokens zero: {info['all_cached_tokens_zero']}")
        rows += [dict(r, run_short=r["run_id"][:16]) for r in run_rows if r["experiment"] == "exp1"]
    if not rows:
        sys.exit("no measurement runs found")
    print("\nAll times in ms. Event spans are on the server's time_ns clock; client TTFT is client monotonic.\n")
    print_table(rows, COLUMNS)
    for r in rows:
        if not r["valid"]:
            print(f"INVALID {r['run_short']} {r['step']}: {r['invalid_reasons']}")
    print("\nMedians over valid repeats (descriptive; individual values above):\n")
    summary = []
    for kind in ("text", "audio"):
        valid = [r for r in rows if r["kind"] == kind and r["valid"]]
        summary.append({"condition": "text_control" if kind == "text" else "audio_030s",
                        "n": len(valid), **{k: median([r[k] for r in valid]) for k in SUMMARY}})
    print_table(summary, [("condition", "condition"), ("n", "n")] + [(k, k) for k in SUMMARY])


if __name__ == "__main__":
    main()
