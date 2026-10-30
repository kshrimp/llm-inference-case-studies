#!/usr/bin/env bash
# Experiment 10: prefill/decode interference; one Modal L4 container per repeat, sequential.
set -euo pipefail
A=projects/long-context-probe/3_fp8_kv_cache_feasibility/app.py
uv run modal run $A::main --stage interference --variant bf16 --repeat 1
uv run modal run $A::main --stage interference --variant bf16 --repeat 2
uv run python projects/long-context-probe/10_prefill_decode_interference/table.py
