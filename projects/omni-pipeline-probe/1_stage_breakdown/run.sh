#!/usr/bin/env bash
# Experiments 1 and 2 share one measurement container (app.py --stage measure):
# server start, warmups, exp 1 (text control / 30 s audio, A B B A), then exp 2
# (10 / 30 / 90 s, 1-2-3-3-2-1). Run from the repository root, one container at a time.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
APP=projects/omni-pipeline-probe/app.py

# 1. No GPU: limits, $2.50 worst case, server command, step order.
uv run python -c "import runpy; m=runpy.run_path('$APP'); m['main'](stage='measure', dry_run=True)"
# 2. Cost gate: nothing running, reconcile spend, then write the cost-log line in docs/STUDY_PLAN.md.
uv run modal app list
uv run modal billing report --start 2026-10-01 --end 2026-11-01 --json > /dev/null
read -r -p "Cost-log reservation written? [y/N] " ok && [ "$ok" = y ]
# 3. One paid H100 container.
uv run modal run "$APP::main" --stage measure
# 4. Tables from every run in results/omni-pipeline-probe/measure/.
uv run python projects/omni-pipeline-probe/1_stage_breakdown/table.py
uv run python projects/omni-pipeline-probe/2_audio_length_validation/table.py
