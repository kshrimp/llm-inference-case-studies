# Experiment 7 findings: long-context retrieval with FP8 KV

**Result: no FP8 quality loss on needle retrieval, and the apparent FP8 advantage was an engine artifact.** First run: saturated FP8 scored 90/90, BF16 84/90. The 7b ablation showed that BF16's 6 misses are deterministic, need **both** radix prefix-cache reuse **and** KV-pool pressure, and are **not dtype-specific**. FP8 with its pool capped to BF16's size fails the same prompts (83/90). The behavior points to a mini-SGLang prefix-cache bug, investigated in [experiment 9](../9_minisgl_prefix_cache_uaf/README.md). Under equal conditions FP8 matches BF16 (83 vs 84 of 90, identical text on 83/90), so long-context retrieval gives **no reason to build v3 per-layer scales**.

Reproduce with `uv run python projects/long-context-probe/7_fp8_long_context_retrieval/table.py`. Raw records are in `results/long-context-probe/fp8-kv-needle/`.

## Runs

The task is 90 prompts: about 1,918 / 3,886 / 7,630 tokens × needle depth 10% / 50% / 90% × 10 trials, under the Qwen3 chat template with thinking disabled, greedy, scored by exact code match. The total prompt length is 402,984 tokens, so the BF16 pool (172,544 tokens) cannot hold every request at once; FP8's 345,344-token pool holds much more of it.

| Run ID | Variant | Prefix cache | Prompts | Pool tokens | Correct | Failed prompts (length@depth#trial) |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `20261025T035055Z-needle-bf16-r1` | BF16 | radix | 90 | 172,544 | **84** | 2048@0.5 #3 #7; 2048@0.9 #1 #2 #3; 8000@0.9 #1 |
| `20261025T035140Z-needle-fp8_sat-r1` | FP8 (sat) | radix | 90 | 345,344 | **90** | none |
| `20261025T035446Z-needle-bf16-r2` | BF16 repeat | radix | 90 | 172,544 | 84 | **identical to r1** (identical text 90/90) |
| `20261025T035532Z-needle-bf16_naive-r2` | BF16 | **naive** | 90 | 172,544 | 89 | 8000@0.1 #7 only |
| `20261025T035619Z-needle-bf16_len2048-r2` | BF16 | radix | **30** (2k only) | 172,544 | 30/30 | none |
| `20261025T035649Z-needle-fp8_sat_capmatch-r2` | FP8, pool = BF16's 674 pages | radix | 90 | 172,544 | **83** | the same 6 as BF16, plus 2048@0.5 #4 |

The r1 pool sizes are from their allocation logs ("Allocating 172544 tokens" and "Allocating 345344 tokens"); from the 7b runs onward the runner records pool size in the JSON.

## What the ablation shows

- **Deterministic:** BF16 r1 and r2 produce byte-identical outputs for all 90 prompts. This is not run-to-run noise.
- **Needs prefix-cache reuse:** with `cache_type="naive"`, all five 2k-token failures disappear (89/90). The prompts share long identical filler prefixes, so the radix cache reuses KV across them.
- **Needs pool pressure:** with radix cache but only the 30 short prompts (about 57,500 tokens, well within the pool), all 30 are correct.
- **Not about dtype:** FP8 at BF16's pool size fails the same prompts. FP8's 90/90 came from its larger pool (less pressure), not from FP8 numerics.
- **Failure signature:** dropped digits ("7531410" → "75310.", "1529296" → "15296") and garbled tokens ("912aysia project project…"), which is what corrupted context looks like rather than a model limit. The one BF16 `naive` miss (8k, 10% depth) has a different pattern and is the only plain model error.

## Quality conclusion for FP8

Under identical scheduling (FP8 capped vs BF16 r1), FP8 scored 83/90 against BF16's 84/90 and produced identical text on 83/90 prompts. Retrieval accuracy at 2k–8k context shows no FP8 penalty. Together with the 180/200 vs 182/200 arithmetic result, there is no evidence of a quality gap that would justify v3 per-layer scales for this model.

## Limits

- One model, 10 trials per cell, synthetic filler haystack, greedy decoding.
- The engine artifact affects these accuracy numbers, so the cleanest FP8 quality comparison is the capacity-matched pair above, or the `naive` cache.
- Whether the failures come from the suspected `cache_req` use-after-free is tested in experiment 9, not here.

## Cost

The experiment 7 and 7b containers (2 + 4) are recorded on the study-plan status board, with actuals from `modal billing report --json`.
