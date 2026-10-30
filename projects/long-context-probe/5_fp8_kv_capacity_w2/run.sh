#!/usr/bin/env bash
# Experiment 5 launch commands (one Modal L4 container each, run sequentially).
# The runner is shared with experiment 3 so both use identical pinned images.
set -euo pipefail
A=projects/long-context-probe/3_fp8_kv_cache_feasibility/app.py
# Full-capacity FP8 vs BF16 (2026-10-24 morning)
uv run modal run $A::main --stage bench --variant bf16 --repeat 1
uv run modal run $A::main --stage bench --variant fp8_sat --repeat 1
uv run modal run $A::main --stage bench --variant fp8_sat --repeat 2
uv run modal run $A::main --stage bench --variant bf16 --repeat 2
# Capacity-matched control: FP8 pool capped at BF16's 674 pages (2026-10-24 evening)
uv run modal run $A::main --stage bench --variant bf16 --repeat 3
uv run modal run $A::main --stage bench --variant fp8_sat_capmatch --repeat 1
uv run modal run $A::main --stage bench --variant fp8_sat_capmatch --repeat 2
uv run modal run $A::main --stage bench --variant bf16 --repeat 4
uv run python projects/long-context-probe/5_fp8_kv_capacity_w2/table.py
