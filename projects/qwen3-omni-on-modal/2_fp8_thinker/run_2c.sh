#!/usr/bin/env bash
# Experiment 2c: one H100 container, ceiling $3.39 (budget.py row 2c). Log the cost line first.
set -euo pipefail
uv run modal app list
uv run modal run projects/qwen3-omni-on-modal/app.py --stage fp8-tune
