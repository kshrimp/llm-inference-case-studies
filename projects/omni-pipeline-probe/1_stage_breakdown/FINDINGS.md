# Experiment 1 findings: text control vs 30 s audio

## What ran

Two independent H100 containers ran the same frozen sequence (see the [project README](../README.md#frozen-settings-for-b-accepted-)): B `20261026T042009Z-e01cd44e-measure` and C `20261026T130358Z-7a636d0b-measure`. Both used sglang-omni `8200f499` with Qwen3-Omni-30B-A3B-Instruct BF16 in text-only mode. The thinker radix cache was off, and every thinker prefill in both server logs shows `#cached-token: 0`. Every request was preceded by a warmup for its path. In each container the order was text, audio, audio, text. Output was greedy and capped at 32 tokens.

The two conditions had the **same thinker prompt length, 409 tokens**:
- `text_control`: "Repeat this text exactly:" plus the first 262 words of the Gutenberg passage, sized in the container by server-counted probes (both containers chose 262 words → 409 tokens).
- `audio_030s`: the transcription instruction plus the 30 s clip (variants v01, v02), also 409 tokens.

**All 8 measured requests are valid.** Each returned exactly 32 tokens with `finish_reason=length`, every audio request's `encoder_cache` trace shows `miss`, and all events are present and in order.

These are **client-observed, single-request localhost latencies plus in-engine request events**. They say nothing about throughput or concurrency. Regenerate every number with `uv run python projects/omni-pipeline-probe/1_stage_breakdown/table.py`; a copy of the output is saved at `results/omni-pipeline-probe/tables/exp1_table.md`.

## Results (ms)

| Run | Condition | Client TTFT | Completion | Preprocess | Audio encoder | Admission → thinker queue | Thinker first forward | Decode ms/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B | text a | 41.5 | 172.3 | 2.1 | — | 4.1 | 25.6 | 4.3 |
| B | audio v01 | 70.6 | 219.0 | 18.4 | 8.9 | 30.7 | 28.2 | 4.8 |
| B | audio v02 | 68.5 | 218.0 | 17.9 | 8.8 | 30.1 | 28.0 | 4.8 |
| B | text b | 39.6 | 169.4 | 1.9 | — | 3.8 | 25.5 | 4.2 |
| C | text a | 40.3 | 171.3 | 1.8 | — | 3.8 | 25.7 | 4.2 |
| C | audio v01 | 66.6 | 212.1 | 15.8 | 8.5 | 27.5 | 28.1 | 4.7 |
| C | audio v02 | 67.2 | 213.9 | 16.3 | 8.8 | 28.1 | 28.1 | 4.7 |
| C | text b | 40.1 | 170.0 | 1.9 | — | 3.9 | 25.7 | 4.2 |
| **Median, text (n=4)** | | **40.2** | **170.6** | 1.9 | — | 3.9 | 25.7 | 4.2 |
| **Median, audio (n=4)** | | **67.8** | **216.0** | 17.1 | 8.8 | 29.1 | 28.1 | 4.8 |

Decode ms/token is the thinker's first-chunk-to-completion span divided by 31. The median gap between successive thinker token chunks agrees: 4.1 ms for text and 4.6 ms for audio.

## Analysis

1. **At the same 409-token thinker prompt, audio reaches first text about 28 ms later (67.8 vs 40.2 ms, +69%).** Almost all of the extra time comes before the thinker starts. Admission → thinker queue is 29.1 ms for audio vs 3.9 ms for text, a difference of 25.2 ms. Within that, preprocessing adds 15.2 ms; it includes audio loading and feature extraction on the CPU. The encoder adds 8.8 ms. Hand-offs and other work make up the rest. The thinker's first forward is 2.4 ms longer for audio (28.1 vs 25.7); these events do not identify the cause of that difference.
2. **The ordering held in all 8 requests across two containers.** Every audio request was slower than every text request on every span listed above. The largest container-to-container shift was about 3 ms in audio preprocessing (B 18.2 vs C 16.1). Medians are descriptive.
3. **Unexpected: audio requests show about 0.5 ms more between successive thinker output chunks than text at the same prompt length** (4.6 vs 4.1 ms median gap; per-request values 4.64–4.77 for audio vs 4.10–4.12 for text across B/C). Both paths use CUDA graphs and one running request. A simple KV-length explanation is unlikely: 10 s audio, with 149 prompt tokens, also has about a 4.6 ms gap (experiment 2). The gap could arise in GPU execution, per-step CPU work, or event delivery; request-level events cannot distinguish them. Over 31 continuation steps, the measured span differs by about 16 ms (216.0 − 170.6 = 45.4 ms total; 27.6 ms of that precedes first text).

**Agent expectation, written before reading B and not the user's:**
- 30 s TTFT of 70–100 ms: observed 66.6–70.6 ms, at the low edge.
- Text reaches first text 30–50 ms sooner: observed 27.6 ms, just below the range.
- Similar thinker first forward for text and audio: confirmed (+2.4 ms).
- Equal thinker output chunk gaps: **wrong**; the audio condition has about 0.5 ms larger gaps. The cause is open.

## Limits

- One duration (30 s), one instruction pair, one prompt length, and sequential single requests, so nothing here measures concurrency, queueing under load, or goodput.
- The text and audio instructions differ ("Repeat this text exactly" vs "Transcribe…"), and both produce text of the same passage. Prompt length is matched; prompt content is not.
- The first 32 generated tokens are similar prose but not identical: the text control begins with printed numbering and formatting, while audio output does not. This mixture-of-experts model can route different token sequences differently. The chunk-gap difference is associated with these two conditions; it is not yet an isolated audio-modality overhead.
- Event spans include CPU scheduling and Python overhead, not GPU time alone. Spans overlap and do not sum to client TTFT.
- About 7 ms of client dispatch → server admission (HTTP plus request parsing) is part of TTFT in both conditions.
- Request recording was on for every measured request. Container A's single off/on pair suggested no large overhead, but that is not measured here.
- n = 4 per condition across 2 containers.

## Next

- **Locate the 0.5 ms thinker chunk-gap difference, if continuing this side probe.** First compare the generated token IDs and read the per-step multimodal path in `models/qwen3_omni/thinker_model_runner.py` and `mrope_positions.py`. Where feasible, construct a text control with matching output tokens. Then take one bounded text/audio torch trace pair (`/start_profile` with `enable_torch`) to compare CPU gaps, GPU kernels, and MoE routing. Profiled timing is diagnostic; any performance claim needs separate unprofiled runs.
- **Preprocessing** is the largest audio-only cost; [experiment 2](../2_audio_length_validation/FINDINGS.md) shows how it scales with duration.
- **Cost and cleanup:** B $0.351583, C $0.319020 (Modal billing `--json`). Both apps are stopped.
