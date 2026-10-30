# Context-length validation

**Current state:** the six unprofiled 1,000-output runs and three 128-output trace runs completed. Read their [findings](FINDINGS.md). The commands in steps 3 and 5 document the completed sequence; they are not a request to rerun it.

This experiment tests the [profiling finding](../1_profile_baseline/FINDINGS.md): with 1,000 output tokens, does a longer starting context increase offline `LLM.generate` time? It uses only unmodified mini-SGLang `9a91cfafe754aa85daee49998176275667eb58f2`. Input lengths are **1,024, 4,096, and 8,000**; output length, seeded token generator, greedy sampling, model, bf16, L4, page size, and CUDA graph setting stay fixed. The largest request is 9,000 total tokens under the configured 9,216-token limit.

**Prediction before these runs:** elapsed time should rise with input length because prefill grows and each decode step can attend to more KV tokens. The existing 8,000/1,000 trace also predicts that decode-attention kernel time will rise across input lengths while matrix-vector kernel time changes less. A flat or noisy result weakens that explanation. Neither trace duration nor whole-call time alone proves memory bandwidth is limiting.

## 1. Local checks; no GPU

Run from the repository root:

```bash
cd /path/to/llm-inference-case-studies
uv run python -m py_compile projects/long-context-probe/2_context_length_validation/app.py projects/long-context-probe/2_context_length_validation/table.py
uv run modal run projects/long-context-probe/2_context_length_validation/app.py::main --help
uv run python -c 'import runpy; m=runpy.run_path("projects/long-context-probe/2_context_length_validation/app.py"); m["main"](input_length=8000, repeat=1, dry_run=True)'
```

Expect help to list `--input-length`, `--repeat`, `--capture-trace`, and `--dry-run`. The dry run should print `"gpu_launched": false`, the baseline commit, and `"estimated_max_usd": 0.34`; it creates no result folder. If `uv` cannot write its cache, prefix commands with `UV_CACHE_DIR=/tmp/uv-long-context-cache`.

## 2. Cost and cleanup gate before **each** paid command

```bash
uv run modal app list
```

Expect no `long-context-length-validation` app still running. Check current Modal Usage and [pricing](https://modal.com/pricing). Enter the next one-container reservation (**up to $0.34**, at the rate checked before the run), spend so far, and remaining approved sprint budget in the study-plan status board (private study notes). Stop if the next run would exceed the remaining approval. Six unprofiled commands reserve **$2.04**; three additional short-trace commands reserve **$1.02**, for **$3.06** total if all are run. Execute them sequentially, never concurrently.

## 3. Two repeats at each length

```bash
uv run modal run projects/long-context-probe/2_context_length_validation/app.py::main --input-length 1024 --repeat 1
uv run modal run projects/long-context-probe/2_context_length_validation/app.py::main --input-length 4096 --repeat 1
uv run modal run projects/long-context-probe/2_context_length_validation/app.py::main --input-length 8000 --repeat 1
uv run modal run projects/long-context-probe/2_context_length_validation/app.py::main --input-length 8000 --repeat 2
uv run modal run projects/long-context-probe/2_context_length_validation/app.py::main --input-length 4096 --repeat 2
uv run modal run projects/long-context-probe/2_context_length_validation/app.py::main --input-length 1024 --repeat 2
```

After **each** command, expect `Saved results/long-context-probe/context-length-validation/<run-id>/run.json`, `"status": "completed"`, `"actual_output_tokens": 1000`, a positive `elapsed_seconds`, and the baseline source SHA. A failure or short output stays as a failed/qualified record; investigate it before proceeding. The 8,000-token result here is a **new** baseline because this runner uses greedy sampling, unlike the first probe's temperature 0.6. Do not compare them as if sampling were held fixed.

## 4. Read results and decide the next measurement

```bash
uv run python projects/long-context-probe/2_context_length_validation/table.py
uv run modal app list
uv run modal billing report --start "$START_DATE" --end "$END_DATE" --json > report.json # --csv wraps at 80 columns when piped
```

Expect nine rows with separate repeats, lengths, output counts and hashes, elapsed seconds, and source commit. The six unprofiled runs predate the `measurement` field, so the table shows `unknown` for that field on those records; their recorded 1,000-token counts establish the unprofiled workload. The three newer rows say `trace_128`. Check that output hashes match within each length for the unprofiled repeats; compare repeat spread with the difference among lengths. The billing report has app-level costs, not per-run charges. A later export posted all six original validation charges; refresh again for the three trace-run charges.

## 5. Reproduce the decode-attention comparison from three short traces

This completed step addresses the phase ambiguity in whole-call time. Each trace holds generation at **128 tokens** and varies only the initial input length. The profiler changes timing, so compare **kernel patterns and event durations** across traces, not their `elapsed_seconds` against the unprofiled runs. These launch commands are retained for reproduction; a new run requires the cost gate again.

```bash
uv run modal run projects/long-context-probe/2_context_length_validation/app.py::main --input-length 1024 --repeat 1 --capture-trace
uv run modal run projects/long-context-probe/2_context_length_validation/app.py::main --input-length 4096 --repeat 1 --capture-trace
uv run modal run projects/long-context-probe/2_context_length_validation/app.py::main --input-length 8000 --repeat 1 --capture-trace
```

Expect `"measurement": "trace_128"`, 128 actual output tokens, and a `Download trace:` line from each successful run. Execute each printed `uv run modal volume get ...` command from the repository root; it downloads `trace.json` into that run's result folder. The remote Volume is `long-context-length-traces`; `uv run modal volume ls long-context-length-traces` lists the exact filenames. If a trace is absent, inspect the saved `run.json` and Modal logs before retrying.

For each downloaded trace, substitute its exact path in this command:

```bash
uv run python projects/long-context-probe/1_profile_baseline/decode_step_trend.py results/long-context-probe/context-length-validation/YOUR_TRACE_RUN_ID/trace.json --window 32
```

Expect `decode_steps_inferred` near 127 (the first generated token follows prefill), 28 decode-attention kernels per step, and first/last 32-step medians for attention, the main matrix-vector kernel, and the interval between attention starts. Compare the **first 32-step attention median** at 1,024, 4,096, and 8,000 input tokens. If event counts do not match the grouping assumptions, the script stops; inspect the trace rather than forcing a conclusion.

The [findings](FINDINGS.md#matched-128-output-chrome-traces) record the observed values and exact trace IDs. The existing 8,000/1,000 trace shows an early-to-late attention trend, but it used temperature 0.6; the three new traces provide a matched greedy-generation comparison. Run the table and cleanup commands again after any reproduction, and reconcile posted app-level spend.

The trace script infers steps from kernel counts and timestamp order; it does not record an explicit scheduler phase or each request's exact KV length. Its per-step kernel sums may overlap with other work. A rising attention trend is stronger evidence for KV-dependent attention cost than whole-call time, but still does not prove that FP8 KV would improve speed. If attribution remains ambiguous, add explicit phase and per-step markers only in the external mini-SGLang clone and save its diff. The [FP8 KV-cache feasibility study](../3_fp8_kv_cache_feasibility/README.md) starts only after the user's source map and CPU toy engine.
