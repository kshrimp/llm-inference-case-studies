#!/usr/bin/env bash
# Experiment 3: speech output on the colocated one-H100 BF16 server (app.py speech-* stages).
# Run from the repository root, one container at a time; write the cost-log line before each launch.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
APP=projects/omni-pipeline-probe/app.py
dry() { uv run python -c "import runpy; m=runpy.run_path('$APP'); m['main'](stage='$1', dry_run=True, lever='${2:-}')"; }
gate() {
  uv run modal app list
  uv run modal billing report --start 2026-10-01 --end 2026-11-01 --json > /dev/null
  read -r -p "Cost-log reservation written? [y/N] " ok && [ "$ok" = y ]
}
# Step 0′ (CPU): config + lever resolution.
uv run modal run "$APP::main" --stage speech-cpu
# A′ smoke gate ($3.31 worst case), then the report; B′ only if GATE FOR B′ passes.
dry speech-smoke; gate; uv run modal run "$APP::main" --stage speech-smoke
uv run python projects/omni-pipeline-probe/speech_report.py "$(ls -td results/omni-pipeline-probe/speech/*speech-smoke | head -1)"
# B′ baseline + torch trace ($2.90).
dry speech-measure; gate; uv run modal run "$APP::main" --stage speech-measure
uv run python projects/omni-pipeline-probe/speech_report.py "$(ls -td results/omni-pipeline-probe/speech/*speech-measure | head -1)"
# C′ ABA with the lever chosen from B′ ($4.11).
LEVER=${LEVER:-L1}
dry speech-ab "$LEVER"; gate; uv run modal run "$APP::main" --stage speech-ab --lever "$LEVER"
uv run python projects/omni-pipeline-probe/speech_report.py "$(ls -td results/omni-pipeline-probe/speech/*speech-ab-"$LEVER" | head -1)"
uv run python projects/omni-pipeline-probe/3_speech_output/table.py
