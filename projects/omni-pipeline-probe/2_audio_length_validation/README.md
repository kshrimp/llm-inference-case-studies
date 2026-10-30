# Experiment 2: audio-duration validation

**Status:** conditional. Run after [experiment 1](../1_stage_breakdown/README.md) only if the audio path, request correlation, stage clocks, cache protocol, and artifact validation work. Settings frozen 2026-10-26 ([project README](../README.md#frozen-settings-for-b-accepted-2026-10-26)); no user prediction (waived).

## Question

When the same audio-in, text-out request uses longer WAV input, how do client time to first text, audio encoder interval, and supported thinker intervals change? Does a stage that dominates the shorter request still dominate the longer one?

## Conditions

Frozen after container A: the nested 10 / 30 / 90 s clips (149 / 409 / 1,189 thinker tokens, verified in A), the transcription instruction, greedy, `max_tokens: 32` (valid only with 32 tokens and `finish_reason=length`); warmed per duration with variant v07; measured variants v01–v04, each used once per container. Keep the text instruction, sampling, output cap, client placement, server settings, and cache protocol fixed. Record SHA-256, sample count, duration, raw `usage`, and actual thinker positions if available. Do not treat a nominal audio duration or API prompt count as a verified thinker token count.

Each request uses its own manifest variant of the clip and counts only if the `encoder_cache` trace shows `action=miss` for it. Run two repeats per duration, sequentially in a balanced order: short, medium, long, long, medium, short. Reset or disable prefix caching before every measured request. Reuse experiment 1's `audio_mid` records only if their exact input, settings, profile state, cache state, and clock contract match; otherwise retain them as separate observations. Record all failures and rejected long clips rather than silently replacing a condition.

## Analysis

Show each request's client TTFT and completion latency, actual output count, supported event intervals, duration, and thinker positions. Plot or tabulate each metric against **audio seconds**, with individual repeats visible. With only three lengths and two repeats, report the measured change and possible step changes; a straight-line slope may be descriptive but is not a fitted scaling law. Distinguish audio feature extraction, encoder, thinker queue, first-output path, and continuation only where the pinned events expose them. A growing client TTFT without a growing recorded encoder interval points to another stage or unmeasured gap; inspect raw events before explaining it.

If actual output counts vary, compare TTFT and pre-output spans first; completion latency and a decode proxy need matching counts or an explicit limitation. The prior Qwen3-0.6B/L4 study is context for a hypothesis, not a quantitative cross-model treatment comparison. No claim about GPU kernel bottlenecks follows from request-level events alone.

## Stop rules

Stop this experiment if the longest clip is rejected or truncated, event clocks/request IDs cannot be reconciled, or cache state cannot be controlled. Preserve a qualified short/medium result if valid. A later 8,000-prompt / 1,000-output extension requires duration and memory checks, matched controls, and a separate cost reservation.
