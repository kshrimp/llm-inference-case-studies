#!/usr/bin/env bash
# Experiment 4b: one H100 container, ceiling $7.42 (budget.py row 4b). Log the cost line first.
set -euo pipefail
uv run modal app list
uv run modal run projects/qwen3-omni-on-modal/app.py --stage capacity-sweep
