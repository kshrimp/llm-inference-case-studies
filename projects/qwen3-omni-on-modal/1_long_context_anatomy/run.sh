#!/usr/bin/env bash
# Experiment 1: exact commands (run from the repo root). Each `anatomy` call is one
# H100 container, ceiling $3.39 (budget.py row 1). Log the cost line first.
set -euo pipefail
uv run modal app list
uv run modal run projects/qwen3-omni-on-modal/app.py --stage anatomy   # container 1
uv run modal run projects/qwen3-omni-on-modal/app.py --stage anatomy   # container 2 (independent repeat)
for r in results/qwen3-omni-on-modal/1_anatomy/*-anatomy; do
  python projects/qwen3-omni-on-modal/1_long_context_anatomy/table.py "$r"
  python projects/qwen3-omni-on-modal/1_long_context_anatomy/analyze_trace.py "$r"/traces/*/*.gz
done
