"""Experiment 3 tables: per-session medians for every speech-measure / speech-ab run.

Reads results/omni-pipeline-probe/speech/*/report.json (run speech_report.py first) and
prints, per run and server session, the measured requests' median and range of TTFT,
TTFA, RTF, thinker ms/token and talker ms/frame for each condition. Individual rows
are in each report.json.

Usage: uv run python projects/omni-pipeline-probe/3_speech_output/table.py
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("results/omni-pipeline-probe/speech")
KEYS = ("ttft_ms", "ttfa_ms", "completion_ms", "rtf", "min_slack_ms", "thinker_ms_per_token",
        "talker_ms_per_frame")


def cell(vals: list[float]) -> str:
    if not vals:
        return "—"
    return f"{statistics.median(vals):.3g} [{min(vals):.3g}–{max(vals):.3g}]"


def main() -> None:
    print("| Run | Session | Condition | n | " + " | ".join(KEYS) + " |")
    print("| --- | --- | --- | ---: |" + " --- |" * len(KEYS))
    for report in sorted(ROOT.glob("*/report.json")):
        rep = json.loads(report.read_text())
        if rep["stage"] == "speech-smoke":
            continue
        for name, sess in rep["sessions"].items():
            groups = defaultdict(list)
            for r in sess["rows"]:
                if r["measured"] and r["status"] == "completed":
                    groups[r["condition"]].append(r)
            for cond, rows in sorted(groups.items()):
                cells = [cell([r[k] for r in rows if r[k] is not None]) for k in KEYS]
                print(f"| {rep['run_id'][:24]} | {name} | {cond} | {len(rows)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
