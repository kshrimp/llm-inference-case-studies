#!/usr/bin/env bash
# Experiment 2: exact commands (repo root). 2a = one H100 container, ceiling $3.39 (budget.py row 2a).
set -euo pipefail
uv run modal app list
uv run modal run projects/qwen3-omni-on-modal/app.py --stage fp8-smoke
python projects/qwen3-omni-on-modal/2_fp8_thinker/table.py
# 2b-1 and 2b-2 (one H100 container each, ceiling $3.39 each)
uv run modal run projects/qwen3-omni-on-modal/app.py --stage fp8-backends
uv run modal run projects/qwen3-omni-on-modal/app.py --stage fp8-combined
python projects/qwen3-omni-on-modal/2_fp8_thinker/table_2b.py
for t in results/qwen3-omni-on-modal/2b*/*/traces/*/*.trace.json.gz; do
  python projects/qwen3-omni-on-modal/slim_trace.py "$t" "$(dirname "$t")/slim.json.gz"
  python projects/qwen3-omni-on-modal/1_long_context_anatomy/analyze_trace.py "$(dirname "$t")/slim.json.gz"
done
