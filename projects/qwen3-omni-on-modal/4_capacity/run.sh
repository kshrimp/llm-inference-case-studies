#!/usr/bin/env bash
# Experiment 4a: one H100 container, ceiling $3.39 (budget.py row 4a). Log the cost line first.
set -euo pipefail
uv run modal app list
uv run modal run projects/qwen3-omni-on-modal/app.py --stage capacity-scan
python projects/qwen3-omni-on-modal/4_capacity/table.py
