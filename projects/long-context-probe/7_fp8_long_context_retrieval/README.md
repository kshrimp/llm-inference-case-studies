# Experiment 7: long-context retrieval with FP8 KV

**Question.** The arithmetic check (FP8 v2 180/200 vs BF16 182/200) uses short contexts, so it is weakly sensitive to KV error. Does saturated FP8 lose facts stored deep in a long context? This decides whether v3 per-layer scales are worth building.

## Why we ran this

All FP8 quality evidence so far came from short contexts: 8 fixtures and 200 arithmetic problems. FP8 rounding error lives in the stored K/V, and the attention softmax has to pick the right key out of more of them as context grows, so long-context retrieval is where an error would show first. The answer decides the next engineering step. A gap would justify v3 (calibrated per-layer scales); no gap would mean the clamp is enough for this model. Needle-in-a-haystack is cheap, deterministic, and scored automatically, so it fits a single paid container per variant.

## What we did

1. Built 90 needle prompts (3 lengths × 3 depths × 10 trials) and checked them on CPU with the pinned tokenizer before any paid run.
2. **Experiment 7:** ran all 90 on BF16 and on saturated FP8. FP8 scored 90/90, BF16 only 84/90, a result that made no sense as a dtype effect.
3. **7b ablation:** to find the cause of BF16's misses, ran a BF16 repeat (is it deterministic?), BF16 without prefix-cache reuse (`bf16_naive`), BF16 on only the 2k prompts (no pool pressure), and FP8 with its pool capped at BF16's size (`fp8_sat_capmatch`). The misses needed both prefix-cache reuse and pool pressure, and capped FP8 failed the same prompts. That pointed to an engine bug, which experiment 9 then investigated.

**Design.** The runner is `--stage needle` in [experiment 3's app.py](../3_fp8_kv_cache_feasibility/app.py), with the same images as experiment 6.

- **Haystack:** classic filler ("The grass is green. The sky is blue. …") with one needle, "The secret code for project {name} is {7-digit code}.", at depth 10%, 50%, or 90%. The question at the end asks for that project's code.
- **Lengths:** about 1,918, 3,886, and 7,630 prompt tokens (targets 2k, 4k, 8k), with 10 trials per length × depth, for 90 prompts in total (seed 20261025).
- Qwen3 chat template with thinking disabled; greedy decoding with at most 16 tokens. Scoring: the exact code string appears in the output.
- mini-SGLang prefill is paged, so it reads the prompt's K/V back from the FP8 cache. FP8 error therefore affects prefill attention as well as decode.
- One container per variant: BF16 and fp8_sat. The test is deterministic, so no repeats.

**Checked locally on CPU (2026-10-24)** with the pinned Qwen3 tokenizer: 90 prompts, all under the 9,216-token engine limit, the 50%-depth needle at 0.49 of the text, and thinking disabled in the template.

**Expected signal (agent-written).** BF16 retrieval should be high at 2k and may fall at 8k for a 0.6B model. If FP8 matches BF16 within a few prompts, v3 is not needed for this model. A depth- or length-specific FP8 gap would justify v3.

**7b ablation (added 2026-10-25 after BF16 missed 6 prompts that FP8 got right).** Four more containers isolate the cause: a BF16 repeat (determinism), BF16 with `cache_type="naive"` (no prefix-cache reuse), BF16 on only the 30 ~2k prompts (no pool pressure), and FP8 with its pool capped at BF16's 674 pages (dtype vs admission pattern). These are `NEEDLE_ABLATIONS` in the runner.

## How to run

Run from the repository root. `bash projects/long-context-probe/7_fp8_long_context_retrieval/run.sh` runs everything below in order.

```bash
cd /path/to/llm-inference-case-studies
A=projects/long-context-probe/3_fp8_kv_cache_feasibility/app.py
# 7: BF16 vs saturated FP8, all 90 prompts, radix prefix cache (engine default)
uv run modal run $A::main --stage needle --variant bf16 --repeat 1
uv run modal run $A::main --stage needle --variant fp8_sat --repeat 1
# 7b: repeat / no prefix cache / no pool pressure / FP8 at BF16 pool size
uv run modal run $A::main --stage needle --variant bf16 --repeat 2
uv run modal run $A::main --stage needle --variant bf16_naive --repeat 2
uv run modal run $A::main --stage needle --variant bf16_len2048 --repeat 2
uv run modal run $A::main --stage needle --variant fp8_sat_capmatch --repeat 2
uv run python projects/long-context-probe/7_fp8_long_context_retrieval/table.py
uv run modal app list
```

## Files

- [`run.sh`](run.sh): experiment 7 and the 7b ablation, in order.
- [`table.py`](table.py): per-run accuracy by length × depth, failing prompts, and paired comparisons against the first BF16 run.
- Runner: `--stage needle` in [experiment 3's app.py](../3_fp8_kv_cache_feasibility/app.py) (`needle_prompts`, `needle_case`, `NEEDLE_ABLATIONS`).
- Results and analysis: [FINDINGS.md](FINDINGS.md).

Cost: 2 + 4 containers × $0.34 worst case. Results go to `results/long-context-probe/fp8-kv-needle/`.
