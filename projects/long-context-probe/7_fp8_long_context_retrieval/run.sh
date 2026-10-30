#!/usr/bin/env bash
# Experiment 7 (needle retrieval) and 7b (BF16-failure ablation); one Modal L4 container each, sequential.
set -euo pipefail
A=projects/long-context-probe/3_fp8_kv_cache_feasibility/app.py
# 7: BF16 vs saturated FP8, all 90 prompts, radix prefix cache (engine default)
uv run modal run $A::main --stage needle --variant bf16 --repeat 1
uv run modal run $A::main --stage needle --variant fp8_sat --repeat 1
# 7b: why did BF16 miss 6 prompts? repeat / no prefix cache / no pool pressure / FP8 at BF16 pool size
uv run modal run $A::main --stage needle --variant bf16 --repeat 2
uv run modal run $A::main --stage needle --variant bf16_naive --repeat 2
uv run modal run $A::main --stage needle --variant bf16_len2048 --repeat 2
uv run modal run $A::main --stage needle --variant fp8_sat_capmatch --repeat 2
uv run python projects/long-context-probe/7_fp8_long_context_retrieval/table.py
