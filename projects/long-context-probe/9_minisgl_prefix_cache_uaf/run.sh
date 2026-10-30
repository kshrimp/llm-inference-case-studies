#!/usr/bin/env bash
# Experiment 9: unit test without/with the fix (CPU), then detector and fixed-engine needle runs (GPU). Sequential.
set -euo pipefail
A=projects/long-context-probe/3_fp8_kv_cache_feasibility/app.py
uv run modal run $A::main --stage unittest --variant bf16_testonly
uv run modal run $A::main --stage unittest --variant bf16_cachefix
uv run modal run $A::main --stage needle --variant bf16_uafdetect --repeat 1
uv run modal run $A::main --stage needle --variant bf16_cachefix --repeat 1
uv run python projects/long-context-probe/9_minisgl_prefix_cache_uaf/table.py
