# Experiment 2 findings: audio duration 10 / 30 / 90 s, 2026-10-26

## What ran

The same two containers as experiment 1, B `20261026T042009Z-e01cd44e-measure` and C `20261026T130358Z-7a636d0b-measure`, after experiment 1's requests. Each duration was warmed with variant v07. Measured variants v01–v04 were each used once per container, in the order 10, 30, 90, 90, 30, 10 s. Output was greedy and capped at 32 tokens. **All 12 measured requests are valid:** each returned 32 tokens with `finish_reason=length`, every `encoder_cache` trace shows `miss`, every log shows `#cached-token: 0`, and all events are in order. Thinker prompt lengths were 149 / 409 / 1,189 tokens, exactly 13 audio tokens per second plus a fixed text part.

These are client-observed single-request localhost latencies plus in-engine events. Regenerate the numbers with `uv run python projects/omni-pipeline-probe/2_audio_length_validation/table.py`; a copy of the output is saved at `results/omni-pipeline-probe/tables/exp2_table.md`.

## Results (ms; individual repeats in the table output)

| Audio | Prompt tok | n | Client TTFT | Completion | Preprocess | Audio encoder | Admission → thinker queue | Thinker first forward | Decode ms/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 s | 149 | 4 | 52.6 (51.7–57.9) | 198.7 | 7.8 | 7.0 | 17.6 | 24.5 | 4.7 |
| 30 s | 409 | 4 | 66.3 (65.8–70.2) | 214.2 | 16.7 | 8.2 | 27.9 | 28.1 | 4.8 |
| 90 s | 1,189 | 4 | 118.5 (102.3–121.1) | 268.4 | 53.3 | 14.1 | 71.3 | 35.5 | 4.9 |
| **90 − 10 s** | +1,040 | | **+65.9** | +69.7 | **+45.5** | +7.2 | +53.7 | +10.9 | +0.1 |

Values are medians, with the range for TTFT. The median thinker chunk gap is 4.6 / 4.6 / 4.7 ms.

## Analysis

1. **From 10 to 90 s of audio, client TTFT rises by 66 ms (52.6 → 118.5), and CPU preprocessing accounts for about 69% of the rise (+45.5 ms).** Preprocessing grows roughly linearly at about 0.57 ms per second of audio: 7.8, 16.7, 53.3 ms. The audio encoder grows weakly, by +7.2 ms (7.0 → 14.1), and the thinker's first forward over the longer prompt grows by +10.9 ms (24.5 → 35.5, about 10 µs per extra prompt token). At 90 s, preprocessing alone is 45% of TTFT. The ordering held in both containers: C's second 90 s request had the lowest preprocessing time (40.2 ms) but still the same ranking.
2. **The encoder is not the bottleneck at these lengths.** Warmed, it takes 7–14 ms. Container A's 24.6 ms for its first 90 s request was a first-of-shape cost, which per-duration warmup removed.
3. **Thinker output chunk gaps change little with audio length:** about 4.6–4.7 ms from 149 to 1,189 prompt tokens. The model's large active weight set relative to roughly 0.11 GB of KV at 1,200 tokens makes weight traffic a plausible explanation, but these event spans cannot establish a GPU bottleneck. The dense Qwen3-0.6B [long-context probe](../../long-context-probe/2_context_length_validation/FINDINGS.md) showed a stronger context trend; that comparison is qualitative because the models, GPU and engine differ. Completion time tracks TTFT over these conditions: completion rose 69.7 ms against TTFT's 65.9 ms.
4. **Audio requests have a roughly 0.5 ms larger thinker chunk gap than the text control** (see [experiment 1](../1_stage_breakdown/FINDINGS.md#analysis)). It stays similar across the three audio durations, suggesting a modality-associated per-step cost or gap. Its mechanism is unmeasured.

**Agent expectation, written before reading B and not the user's:**
- Preprocessing grows roughly linearly and dominates the TTFT change: **confirmed** (+45.5 of +65.9 ms).
- The warmed encoder grows weakly: confirmed (+7.2 ms).
- A's 90 s encoder time was a first-of-shape artifact: confirmed (14.1 vs 24.6 ms).

## Limits

- Three durations with four requests each is descriptive. A straight line through 10 / 30 / 90 s is not a scaling law, and longer audio (the 8,000-token, roughly 10-minute case) is untested.
- Each container ran one server, requests were sequential, and preprocessing used the default serial executor with `OMP_NUM_THREADS` equal to the CPU count (8 cores requested). The preprocessing cost could change with more cores, `--preprocessing.factory.max_concurrency`, or concurrent load. None of that was measured.
- The audio is real speech from one reader; other sample rates would add resampling inside preprocessing.

## Next

- **Customer framing (FDE):** in this single-request setup, time to first text grew about 0.8 ms per extra second of audio (65.9 ms over 80 s); 0.57 ms/s of the rise appeared in CPU preprocessing. These are local, descriptive slopes. An optimization decision would need a stated customer workload and a controlled unprofiled test; more CPU cores or parallel preprocessing have not been shown to help a single request.
- **Decode penalty for audio requests:** see experiment 1's next steps (one torch trace pair).
- **Cost and cleanup:** shared with experiment 1, B $0.351583 and C $0.319020; both apps are stopped.
