# Long-context mini-SGLang probe

This separate project runs one synthetic **offline** request with exactly 8,000 input token IDs and up to 1,000 output tokens on one Modal L4. It helps find a performance question to investigate. It does not measure serving latency or establish an upstream regression.

The first four runs and trace interpretation are recorded in [FINDINGS.md](FINDINGS.md). Use the steps below for a new run.
The [project index](../README.md) links the later validation and source-tweak experiments.

Mini-SGLang is pinned to `9a91cfafe754aa85daee49998176275667eb58f2` (MIT). The model is `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`, bf16. The prompt is reproducible random token IDs, not natural language. The app uses existing `llm-bench-hf-cache` and `tiny-engine-jit-cache` Modal Volumes; the pinned model snapshot must already be present. It does not download weights.

## Run sequence

Run all commands from the repository root, **one at a time**. Each `modal run` starts a separate paid GPU container. Local results go to `results/long-context-probe/<run-id>/run.json` and are git-ignored.

### 1. Check the local command (no GPU)

```bash
cd /path/to/llm-inference-case-studies
uv sync
uv run python -m py_compile projects/long-context-probe/1_profile_baseline/app.py
uv run modal run projects/long-context-probe/1_profile_baseline/app.py::main --help
```

Expect the last command to show `--mode TEXT`. These checks do not launch the experiment. If `uv` cannot write its cache in this environment, prefix commands with `UV_CACHE_DIR=/tmp/uv-long-context-cache`.

### 2. Check spend before each paid run

```bash
uv run modal app list
```

Expect no `long-context-probe` app still running. Check actual spend on the Modal Usage page, then add a line to the study-plan status board (private study notes) **before each paid run**: mode, one container, up to $0.34 estimated, spend so far, and remaining approved budget. The existing $10 sprint approval is shared with the original study. Check [current Modal prices](https://modal.com/pricing). Stop if the next run would exceed the remaining approved amount.

### 3. Smoke: does 8,000 input fit?

```bash
uv run modal run projects/long-context-probe/1_profile_baseline/app.py::main --mode smoke
```

Expect `Saved results/long-context-probe/<run-id>/run.json` and JSON with `"status": "completed"`, `"input_tokens": 8000`, `"requested_output_tokens": 8`, `"actual_output_tokens": 8`, and `"valid_count": true`. Inspect the local `run.json` and Modal logs if it fails or is qualified. This mode produces **no trace**. Stop before profiling if it fails.

### 4. Short profile: 8,000 input / 32 output

After repeating step 2:

```bash
uv run modal run projects/long-context-probe/1_profile_baseline/app.py::main --mode profile
```

Expect a saved `run.json`, output count 32, and a line beginning `Download trace:`. The trace is saved on Modal Volume `long-context-probe-traces`; it is **not downloaded automatically**. Copy and execute the complete printed command. It looks like this, with your run ID replacing the example:

```bash
uv run modal volume get long-context-probe-traces 20261023T120000Z-ab12cd34-profile.json results/long-context-probe/20261023T120000Z-ab12cd34-profile/trace.json
```

Expect a local `trace.json` in that run's folder. To see the actual remote filename, run `uv run modal volume ls long-context-probe-traces`; use that name, not the example. If `modal volume get` says “No such file or directory,” confirm the profile completed, the filename appears in `volume ls`, and the destination folder printed by `Saved .../run.json` exists. Open `chrome://tracing` in Chrome, choose **Load**, then select `trace.json`.

This trace includes the 8,000-token prefill and only the first 32 decode tokens. Profiler overhead affects its timing.

### 5. Full run: 8,000 input / 1,000 output, without profiler

After repeating step 2:

```bash
uv run modal run projects/long-context-probe/1_profile_baseline/app.py::main --mode full
```

Expect `"requested_output_tokens": 1000`, `"actual_output_tokens": 1000`, and `"valid_count": true` in `run.json`. `elapsed_seconds` is wall time for one offline `LLM.generate` call after a small warmup; it is neither client latency nor per-token latency. A single run is exploratory. Repeat this mode once if it succeeds and budget remains, keeping both `run.json` files separate.

**This mode does not produce a trace. A trace cannot be retrieved from a completed unprofiled run.**

### 6. Optional full trace: 8,000 input / 1,000 output

If you need a trace covering all 1,000 output steps, repeat step 2 and launch another paid run:

```bash
uv run modal run projects/long-context-probe/1_profile_baseline/app.py::main --mode full-profile
```

Expect output count 1,000 and a `Download trace:` line on success. Execute that printed command as in step 4. The trace may be large, and tracing every step may hit the 8 GiB system-memory or 15-minute execution limit. If it fails, keep the failure record and use the short trace or narrower instrumentation. Do not compare its `elapsed_seconds` directly with unprofiled `full` as an engine performance difference.

### 7. Close the run

```bash
uv run modal app list
```

Expect no `long-context-probe` app still running. Enter actual spend from Modal Usage and the cleanup result on the status board. Inspect the trace for a specific hot span, its time denominator, and whether it is CPU or GPU work. Check the pinned source, write an alternative explanation, and plan a targeted validation. A hot CUDA operation alone does not prove KV-cache pressure; FP8 KV requires memory and correctness evidence, while kernel fusion requires evidence of launch or intermediate-memory cost. Reproduce any proposed issue in the intended upstream engine before proposing a PR.

## Cost and limits

At [Modal's listed prices checked 2026-10-23](https://modal.com/pricing), one L4 is $0.000222/s, CPU is $0.0000131/core/s, and memory is $0.00000222/GiB/s. This app requests one L4, 2 CPU cores, and 8 GiB of system RAM, totaling about **$0.016 per billable minute**. It sets 5 minutes startup, 15 minutes execution, and a 1-minute idle window. The planned maximum is `21 × $0.016 ≈ $0.335`, rounded up to **$0.34 per run**. Three initial modes reserve up to **$1.02**; adding `full-profile` reserves another **$0.34**. Actual charges depend on elapsed billable time and may include image build, Volume storage, downloads, and retries. Record observed charges separately from estimates.

The 15-minute execution timeout may end a slow 1,000-token decode. Each command permits at most one GPU container. Keep container cold start, model load, profiler export, and the timed `generate` call separate. PyTorch profiler operator totals may overlap and cannot be summed into wall time.
