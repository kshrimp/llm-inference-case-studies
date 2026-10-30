#!/usr/bin/env bash
# Experiment 6 launch commands (one Modal L4 container each, run sequentially; shared runner from exp 3).
set -euo pipefail
A=projects/long-context-probe/3_fp8_kv_cache_feasibility/app.py
uv run modal run $A::main --stage sweep --variant bf16 --repeat 1
uv run modal run $A::main --stage sweep --variant fp8_sat --repeat 1
uv run modal run $A::main --stage sweep --variant fp8_sat --repeat 2
uv run modal run $A::main --stage sweep --variant bf16 --repeat 2
uv run python projects/long-context-probe/6_fp8_break_even/table.py
