# Experiment 8: is the SGLang FP8 MHA overflow still live upstream?

**Question.** [Experiment 4](../4_sglang_fp8_kv_reproduction/FINDINGS.md) reproduced NaN KV writes in SGLang 0.5.9's `MHATokenToKVPool.set_kv_buffer` (a plain `.to(float8_e4m3fn)` with no saturation) and fixed them with a clamp. Before proposing anything upstream, check whether current SGLang main has the same problem, and how that relates to open PR [#41732](https://github.com/sgl-project/sglang/pull/41732).

## Why we ran this

Experiment 4 was run on SGLang 0.5.9, but upstream fixes land on main. A PR or issue that reports a bug main no longer has wastes maintainers' time and hurts credibility, which matters because an upstream PR is a sprint goal. Two things needed checking before posting: whether main's source still has the unsaturated cast, and whether its newer PyTorch behaves differently. The second turned out to be the deciding question. The check was also designed to be cheap: source reading and a CPU bisect, with only two small GPU containers.

## What we did

1. Read SGLang main's KV writer, its tensor-core decode choice, the `saturate_to_fp8_range` docstring, the torch pin, and PR #41732's diff and CI. The source was unchanged, but main pins torch 2.14.1.
2. Bisected PyTorch releases on CPU to find when the BF16 → e4m3fn cast changed from NaN to saturating: between 2.12.0 and 2.13.0.
3. Confirmed the same behavior on an L4 with CUDA wheels for 2.12.0 (NaN) and 2.14.1 (saturates).
4. Rewrote experiment 4's upstream comment draft: SGLang main is not affected at runtime, and the bug is specific to releases that ship torch ≤ 2.12.

**Design (no engine GPU runs).**
1. **Source reading:** SGLang main at `773cb938fe9484a8f7e7e46fb1b1fe96412c59ed` (2026-10-25), covering `memory_pool.py`, `flashinfer_backend.py::should_use_tensor_core`, the `saturate_to_fp8_range` helper and its docstring, and the torch pin in `python/pyproject.toml`. Plus PR #41732's diff, review state, and CI.
2. **CPU bisect:** [`torch_cpu_bisect.sh`](torch_cpu_bisect.sh) installs each torch release into a throwaway venv (CPU wheel) and casts 450 / 464 / 470 / 516 / −516 from BF16 to e4m3fn.
3. **CUDA check:** [`cast_probe_torch_versions.py`](cast_probe_torch_versions.py) runs the same cast on an L4 with torch 2.12.0 and 2.14.1 (cu130 wheels), contiguous, strided, and float32-source. It uses two small containers in sequence, with a $0.21 worst-case reservation each.

## How to run

Run from the repository root. The bisect uses throwaway venvs (an optional first argument sets the work directory) and does not touch the project venv. The Modal probe runs two small L4 containers in sequence.

```bash
cd /path/to/llm-inference-case-studies
bash projects/long-context-probe/8_sglang_fp8_upstream_check/torch_cpu_bisect.sh
uv run modal run projects/long-context-probe/8_sglang_fp8_upstream_check/cast_probe_torch_versions.py
uv run modal app list
```

Results go to `results/long-context-probe/sglang-fp8-kv-reproduction/cast_probe_torch_versions.json`; that output path is kept next to experiment 4's `cast_probe.json`.

## Files

- [`torch_cpu_bisect.sh`](torch_cpu_bisect.sh) and [`cast_probe_torch_versions.py`](cast_probe_torch_versions.py): the two checks above.
- [`FINDINGS.md`](FINDINGS.md): results and the revised upstream position.
- The upstream comment draft lives in experiment 4: `UPSTREAM_COMMENT_DRAFT.md` (private draft). It was rewritten after this experiment.
