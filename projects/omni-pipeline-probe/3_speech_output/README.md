# Experiment 3 (optional): diagnose speech-output latency and test one lever

**Status (2026-10-28):** done through C′. **L1 (talker partial start) wins:** `S32` TTFA ~340 → 226 ms and `Sfull` ~650 → 223 ms, with identical text and audio and no stalls. The thinker decodes 41% slower while the two overlap. See [FINDINGS.md](FINDINGS.md). The optional D′ (L1 + MPS) is awaiting a decision. The one-H100 BF16 speech baseline was a feasibility question, not an established configuration: [upstream issue #432](https://github.com/sgl-project/sglang-omni/issues/432) says full BF16 colocation did not fit on an H100, and [PR #454](https://github.com/sgl-project/sglang-omni/pull/454) added a supported FP8 path. Step 0′ must settle the checkpoint/configuration before A′ is launched. Source facts below were read at the pinned `8200f499` checkout; runtime behavior remains to be verified. Every launch needs a cost-log reservation first. No user prediction (waived for this probe). An agent expectation is written before each lever run.

## Step 0′ results (2026-10-27)

**BF16 feasibility.** Issue #432 (opened 2026-05-12, closed 2026-05-24 by PR #454) states in its motivation that full BF16 colocation did not fit on one H100. It is a feature request, not a measurement at our pin. The pinned `8200f499` (2026-10-21) has `examples/configs/qwen3_omni_colocated_h100_bf16.yaml`, and upstream CI uses it for the speech-output TTS test: `tests/test_model/conftest.py::start_qwen3_omni_bf16_colocated_router` starts "2 BF16 colocated replicas (one per H100)" with `audio_output=True` on `[self-hosted, h100]` runners. The checkpoint's safetensors headers at our revision give BF16 sizes: thinker LM + lm_head 61.06 GB, talker LM + code predictor + projections 6.64 GB, audio tower 1.30 GB, vision 1.08 GB, code2wav 0.43 GB. Against the config's budgets on an 85.5 GB (79.6 GiB) H100, the thinker's 0.78 leaves about 5.6 GB beyond its weights and the talker's 0.10 about 1.9 GB. That is tight but plausible on paper. **Decision: BF16 A′ goes ahead with this pinned config, and no FP8 fallback.** A failed or out-of-memory startup is a stop result recorded as such. It does not open an FP8 track without a new decision.

**Config and levers (CPU run `20261027T235454Z-9af2b068-speech-cpu`).** The pinned file is as listed above. Base and all three levers resolve, each to exactly the intended changes:

| Lever | Resolved change |
| --- | --- |
| L1 | `stages.talker_ar.factory.enable_partial_start: False -> True` |
| L2 | `stages.code2wav.factory.initial_codec_chunk_frames: None -> 2`, `stages.talker_ar.factory.codec_coalesce_early_frames: 10 -> 12` |
| L3 | `mps: 'off' -> 'auto'` |

**Relay.** The resolved config uses direct CUDA IPC payloads between same-GPU stages, except the audio encoder, which has `disable_direct_cuda_ipc_payload: true` (shm). No NIXL/UCX relay is selected, so the image is unchanged.

**Torch trace route.** `POST /start_profile` takes `{run_id, enable_torch, event_dir, trace_path_template}`. The template is formatted per stage with `{stage}` (`pipeline/stage/runtime.py`), so each process writes its own trace file.

**Runner.** `app.py` now has stages `speech-cpu`, `speech-smoke`, `speech-measure`, and `speech-ab --lever L1|L2|L3`. Each server session saves its folder under `results/omni-pipeline-probe/speech/<run_id>/<session>/`. [speech_report.py](../speech_report.py) prints per-request client metrics, the event timeline before first audio, and the A′ gate. Both were tested locally against a fake SSE server and synthetic events.

## Container A′ results (2026-10-28)

Run `20261027T235833Z-882c4881-speech-smoke`; checks with [speech_report.py](../speech_report.py) (`report.json` in the run folder). The observed GPU was an H100 80GB HBM3. **BF16 colocation fits:** 77.6 of 81.6 GB were used after readiness. The server was ready after 383 s with a cold JIT cache, including upstream speech warmup. All 9 steps completed. Actual cost was $0.676318. **These are compatibility results from single requests, not benchmark numbers.**

**Gate for B′: pass.** Every required check is true:
- every speech request returned decodable 24 kHz audio, with no talker cap hit
- the encoder cache showed miss / hit / miss…
- all 18 thinker prefill lines showed `#cached-token: 0`
- events correlated under the client `request_id` across all 6 process files
- the talker prefill started after thinker `stage_complete` in every speech request
- event frames (Σ `audio_samples` / 1,920) equalled the client's decoded samples exactly
- the text-only request had no talker events

The same input with the same seed produced **byte-identical audio** (informational).

**Report fix (disclosed):** the first report pass failed `frames_match_samples` because it looked up code2wav events with stage `None`, which the emit call passes. The recorded rows carry `stage: "code2wav"`. Only the report changed; raw files are unchanged.

| Step | TTFT | TTFA | Thinker complete | Talker first frame | First window ready | Audio s | Thinker ms/tok | Talker ms/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `s010_v02` (S32) | 59.7 | 321.9 | 254.3 | 268.0 | 314.7 | 9.26 | 6.35 | 5.59 |
| `s030_v01` (S32) | 85.9 | 359.5 | 284.4 | 303.3 | 351.4 | 8.38 | 6.50 | 5.65 |
| `s030_v02_full` (Sfull, 80 tok) | 73.3 | 646.6 | 573.8 | 591.5 | 639.5 | 22.46 | 6.38 | 5.55 |
| `t030_v03` (T32) | 70.5 | — | — | — | — | — | 6.33 | — |

All times are in ms from client dispatch. Smoke observations, to be checked in B′:
1. **The thinker's whole reply sits on the TTFA path.** About 80% of S32's TTFA comes before thinker `stage_complete`. `Sfull` − `S32` = +287 ms ≈ 48 extra tokens × 6 ms. This is the L1 signal.
2. After the thinker, about 10 ms of hand-off and talker prefill come first. Then 10 talker frames take about 47 ms, then code2wav and delivery about 8 ms. L2 could save about 8 × 5.6 ≈ 45 ms.
3. **Speech is about 14× faster than real time** (talker 5.6 ms per 80 ms frame; RTF 0.07). Minimum playback slack was about 716 ms, with no stalls.
4. **Thinker decode is 6.3–6.5 ms/token on the speech server, against 4.2–4.8 in experiments 1–2.** This also holds for the text-only request, when the talker is idle. It was not expected and is a candidate for diagnosis: process count, config (`cuda_graph_backend_prefill: breakable`), or GPU sharing.
5. The speech requests report `finish_reason=stop` at exactly 32 completion tokens, but text-only requests on the same server report `length`. The stream's final reason comes from a different terminal stage, so completion count, not finish reason, is the validity check for `S32`.
6. **MPS starts on Modal.** The `--mps auto` probe session was ready after 132 s (warm JIT). Its single, unwarmed request (TTFA 388 ms) is not a comparison.

## Container B′ results and lever decision (2026-10-28)

Run `20261028T000826Z-0fa799c7-speech-measure`. The server was ready after 328 s with a warm JIT cache. All 7 measured requests are valid: each had an encoder-cache miss, the expected completion count, and frames equal to samples. Actual cost was $0.502370.

| Condition | n | TTFT median [range] | TTFA median [range] | RTF | Thinker ms/tok | Talker ms/frame |
| --- | ---: | --- | --- | ---: | --- | --- |
| `S32` | 3 | 72.2 [71.6–72.3] | **343 [333–344]** | 0.069 | 6.24–6.62 | 5.53–5.56 |
| `Sfull` (80 tok) | 2 | 72.2 [70.8–73.7] | **656 [651–661]** | 0.068 | 6.50–6.58 | 5.45 |
| `T32` | 2 | 72.5 [71.9–73.2] | — | — | 5.99–6.30 | — |

**Where `S32` TTFA goes** (`S32_v01`, 333 ms; event times from dispatch, which partition the path here because the stages run one after another):

| Interval | ms | Share |
| --- | ---: | ---: |
| Dispatch → thinker first token (HTTP, preprocessing, encoder, prefill) | 69 | 21% |
| Thinker decodes the remaining 31 tokens (`stage_complete` at 262) | 193 | 58% |
| Hand-off: thinker done → talker prefill start | 4.5 | 1% |
| Talker prefill → first frame | 10.6 | 3% |
| Talker frames 1–10 (first window ready at 325) | 47 | 14% |
| Code2wav first window decode | 5.5 | 2% |
| Delivery to client | 2.5 | 1% |

**Torch trace (one profiled `S32`, inflated by the profiler: 7.97 ms/token against 6.3 untraced).** The thinker's 32 decode steps each show about 4.2 ms of kernel work, then about 3.6 ms with no thinker kernels before the next step (median period 7.8 ms). That 4.2 ms of kernel work matches the 4.2–4.8 ms/token of experiments 1–2. So the extra thinker time on the speech server is GPU-idle time between steps, not slower kernels. Each step has a blocking D2H copy, two `Device -> Pageable` copies per step in total. The trace records one process, so it cannot show whether another process's work fills those gaps. Finding the cause would need a CPU stack trace; that is a possible follow-up, not this experiment's lever.

**Decision: L1 (talker partial start) for C′.** The thinker's reply is 58% of `S32` TTFA, and `Sfull` adds 313 ms for 48 more tokens. L2 targets only the 47 ms first-window wait. L3 targets GPU sharing that does not happen while partial start is off.

**Win threshold:** the `S32` baseline spread is 11 ms (333–344), so the threshold is max(30 ms, 2 × 11) = **30 ms** off median `S32` TTFA, with all guardrails holding.

**Agent expectation for C′ (written before C′ launched).**
- With L1 the talker should start after 5 thinker tokens, about 95 ms after dispatch. It then generates frames while the thinker is still decoding, on the same GPU.
- `S32` TTFA should drop from about 343 to about **180–230 ms**. `Sfull` should drop by more, from about 656 to about **200–260 ms**, so TTFA mostly stops depending on reply length.
- Cost: thinker ms/token should rise 10–30% while the two overlap. Completion time should change little. Text and audio should stay identical: greedy thinker, fixed talker seed, though the talker's input timing changes.
- No stalls; RTF stays below 0.2.

## Goal

For the fixed 30 s speech clip and transcription instruction, find where time to first audio (TTFA) goes when the server returns text + speech. If the baseline is viable, test one configuration change suggested by the timeline without making playback stutter or changing the output. Report the measured tradeoff, including a null or negative result. This is a transcription-to-speech probe; a broader voice-assistant recommendation would need a spoken-question workload. A possible upstream issue is a follow-up only when the evidence identifies a reproducible engine defect or missing capability; finding one is not an experiment success criterion.

Order of work: **diagnose** (where does TTFA go?), then **pick one lever** the evidence points to, then **A/B test** it, then **explain** the result. This stays single-request, client-observed serving latency plus in-engine events. It does not measure concurrency, speech quality, or offline throughput.

## What the pinned source says

| Question | Answer at `8200f499` | Consequence |
| --- | --- | --- |
| One-GPU speech mode | `sgl-omni serve --config <yaml> --colocate`, with a `Qwen3OmniSpeechColocatedPipelineConfig` file (`--colocate` is rejected with `--text-only`). The image may contain `/opt/sglang-omni/examples/configs/qwen3_omni_colocated_h100_bf16.yaml`; verify the pinned file and resolved settings in Step 0′ | A config file or CI use does not establish that the full BF16 model fits on an H100. The upstream issue reports the opposite. Do not assume A′ can start |
| Memory split | Candidate BF16 config: thinker 0.78, talker_ar 0.10, image/audio encoder 0.02 each, code2wav 0.02 | Verify the selected config's actual budgets and loaded memory; these fractions do not prove fit |
| Process layout | preprocessing, image_encoder, audio_encoder, thinker, decode, and talker_ar are separate processes assigned GPU 0. Code2wav runs inside the talker process on a priority stream (`COLOCATED_SPEECH_PROCESSES`) | Placement does not show which process is using the GPU at a given time. All events still share `time.time_ns()` |
| **When the talker starts** | The colocated config sets `enable_partial_start=False`, so the talker waits until the thinker's whole reply is finished (`talker_scheduler.py::is_request_build_ready`). The 2-GPU default sets it `True` and starts after 5 thinker chunks | **TTFA includes the entire thinker reply.** This is the most likely lever |
| First audio window | The talker sends the first 10 codec frames one at a time, then in groups of 10. Code2wav first decodes at 10 frames (`stream_chunk_size=10`, `initial_codec_chunk_frames=0`). The upstream docs say `initial_codec_chunk_frames=2` with `codec_coalesce_early_frames=12` makes the first window ready at frame 2 | About 8 talker steps of avoidable wait before the first audio |
| GPU sharing | `--mps auto` enables CUDA MPS when 2+ single-GPU processes share a GPU. The default is `off` (`docs/basic_usage/mps_dp.md`) | May reduce time-slicing stalls, mainly when the thinker and talker overlap. May not be allowed inside a Modal container: *verify in A′* |
| Talker sampling | temperature 0.9, top_k 50, repetition penalty 1.05, up to 4096 frames, `seed` from the request. Fields such as `talker_max_new_tokens` override them | Thinker greedy. Talker upstream default with `seed: 1234` and `talker_max_new_tokens: 600`. Seed reproducibility is *verify in A′* |
| Audio frame | 1 codec frame = 1,920 samples at 24 kHz = **80 ms** | Real time needs > 12.5 frames per second of wall time |
| Speech events | code2wav: `code2wav_first_ingest`, `_first_window_ready`, `_decode_start` / `_end` (with `audio_samples`), `_first_audio`. Talker: shared `scheduler_*` / `stage_*` events with stage `talker_ar` | Talker event names are candidates until A′ |
| Streamed audio | `delta.audio.data` (base64, `audio.format`, default `wav`) in the SSE chunks of `/v1/chat/completions` with `"modalities": ["text","audio"]` | Client decodes each delta, records arrival time and samples |
| Talker prefix cache | Forced off in `talker_scheduler.py` | Only the thinker flag needed |

## Levers (all config flags, no engine code changes)

| Lever | Flag(s) | Evidence in the baseline that points to it | Expected effect | Risk |
| --- | --- | --- | --- | --- |
| **L1 talker partial start** | `--talker_ar.factory.enable_partial_start true` | The talker prefill starts only after thinker `stage_complete`, and TTFA grows with reply length (`Sfull` − `S32`) | TTFA stops depending on reply length. Expect a saving of about the thinker decode time after its 5th token (≈ 130 ms at 32 tokens, more for longer replies) | The thinker and talker then share the GPU at the same time, so thinker ms/token may rise. A likely reason upstream turned it off for colocation |
| **L2 earlier first window** | `--code2wav.factory.initial_codec_chunk_frames 2` and `--talker_ar.factory.codec_coalesce_early_frames 12` | A large share of TTFA lies between talker first frame and `code2wav_first_window_ready` (≈ 10 talker steps) | Saves about 8 talker steps (8 × talker ms/frame) | A 2-frame window may run without a captured CUDA graph, so the first decode could be slower. Possible audio artefact at the start: listen |
| **L3 MPS** | `--mps auto` | Gaps where no stage owns the GPU, or stage intervals that stretch when two stages are active. Most relevant with L1 | Shorter overlapped intervals. Recovers any thinker slowdown from L1 | The MPS daemon may not start on Modal. Then L3 is dropped (a finding in itself) |
| L4 preprocessing threads (only for long audio) | colocated `OMP_NUM_THREADS=8` default; preprocessing process env | Experiment 2: preprocessing is 69% of the 10→90 s TTFT growth | Small at 30 s (≈ 17 ms of ~0.5 s). Can be profiled on CPU locally for $0 | Out of scope unless L1–L3 are exhausted |

Pick the lever by the evidence, not this list's order. Phase 2 tests at most two levers. L1 + L2 together is allowed as a third arm only if each was tested alone first. Upstream-default levers we did not consider (for example, thinker CUDA graph settings) are added here with their source reference before being tested.

## Workload

30 s clip (409 thinker tokens) with the experiment 1 transcription instruction, greedy thinker, `seed: 1234`, `talker_max_new_tokens: 600` (a request that hits it is invalid).

| Condition | `modalities` | Thinker cap | Purpose |
| --- | --- | --- | --- |
| `S32` | text + audio | `max_tokens: 32`, valid only with 32 completion tokens (speech requests report `stop`, see A′ observation 5) | Main case. Its thinker work matches experiment 1 |
| `Sfull` | text + audio | `max_tokens: 128` (A stopped by itself at 80 tokens, so expect `stop`) | A longer reply shows whether TTFA depends on reply length (the L1 signal) |
| `T32` | text | `max_tokens: 32` | Same server, no talker. Tells whether speech mode changes text TTFT |

A server session runs unmeasured warmups (`S32` v07, `T32` v07, `Sfull` v07), then measured `S32` v01, `T32` v02, `Sfull` v03, `S32` v04, `Sfull` v05, `T32` v06, `S32` v00. Every measured request needs an `encoder_cache` `miss`. Each server session reuses these variants, because a restarted server has an empty encoder cache. *Verify in A′:* the cache is in-process, so a restart empties it.

## Metrics

**Primary:** TTFA (dispatch → first `delta.audio` with ≥ 1 decoded sample).

**Guardrails.** A lever counts as a win only if all of these hold:

| Guardrail | Rule |
| --- | --- |
| No playback stalls | Simulated playback starts at the first audio arrival. Every later chunk arrives before it is due. Report the minimum slack |
| Steady-state RTF | (last − first audio arrival) / (audio duration − first chunk duration) stays < 1 |
| Same text | Thinker text and completion count identical to the baseline arm (greedy) |
| Audio sane | Frames from events × 1,920 = client samples (± the last window). Duration within the baseline's range. The user listens to one `S32` WAV per arm |
| Thinker cost | Thinker decode ms/token reported. If it rises, state by how much |
| Memory | Peak GPU memory reported per arm |

**Diagnostic intervals:** client TTFT and completion. Thinker preprocess, encoder, queue, first forward, decode ms/token. Thinker done → talker prefill start. Talker prefill. Talker first frame → 10th frame, and talker ms/frame against the 80 ms budget. `code2wav_first_window_ready` → `code2wav_first_audio`, and the total of `decode_start → decode_end`. `code2wav_first_audio` → client first audio. Report intervals before first audio in time order with gaps shown. Never add overlapping intervals; stages share one GPU by time-slicing, so busy times are wall time, not GPU time. Usage reports no talker token count; it is null with that reason, and frames come from code2wav events.

**Win threshold (set before Phase 2):** median TTFA drops by more than the larger of 30 ms or 2× the spread of baseline `S32` repeats, with every guardrail holding. Experiments 1–2 showed a few ms run-to-run spread for TTFT, but TTFA spread is unknown until B′.

## Phases and containers

**Step 0′ (CPU, ≈ $0.05; feasibility decision).** Confirm the pinned BF16 config file exists, inspect its stage memory budgets and the checkpoint sizes, and reconcile them with issue #432. A resolved config is not proof of fit. If BF16 cannot plausibly fit, do not spend on an H100 BF16 startup attempt. Either stop experiment 3 with that reason or separately freeze an upstream-supported FP8 checkpoint/config, its provenance and revision, and a new cost estimate before any GPU launch. FP8 speech results would be a separate baseline and would not be compared numerically with experiments 1–2's BF16 text-only results. Do not develop FP8 support again. Once a viable baseline is selected, run `sgl-omni config resolve --config <selected yaml> --colocate --thinker.engine.disable_radix_cache true --show diff`, and once more with each lever's flags. Save every output. A lever whose flags do not resolve, or do not reach the stage (for example code2wav factory args inside the talker process), is dropped or fixed now. Also check which relay the colocated processes use. The image has no UCX build (`app.py`); if the relay needs it, add the upstream Dockerfile step and record the image change. Then write the runner, dry-run it locally, and run no GPU before it passes:
- `app.py` stages `speech-smoke` and `speech-session`. The session stage takes a list of server flag sets and runs one server session per set, sequentially, in one container.
- `stream_chat` collects audio deltas.
- `speech_report.py` checks the A′ gate.
- `3_speech_output/{run.sh, table.py}`; results in `results/omni-pipeline-probe/speech/<run_id>/`; `validate.py` extended to that folder.

**Container A′: smoke gate ($2.90 worst case at the recorded rate; launch only after Step 0′).** Selected, pinned colocated server with these steps:
- `text_short`
- 10 s v01 `S32`, then repeated with the same seed (expect a cache hit; compare the audio SHA-256)
- 10 s v02 `S32`
- 30 s v01 `S32`
- 30 s v02 `Sfull`
- 30 s v03 `T32` (expect no talker events)
- profile off/on pair (10 s v03/v04)
- at the end, one `--mps auto` restart just to see whether the daemon starts

Gate (all true):
- Observed GPU is an H100 80 GB, and the server is healthy.
- Every step completes.
- Every speech request returns decodable 24 kHz audio, without hitting the talker cap.
- Events correlate under the client `request_id` across all process files.
- The encoder cache shows miss / hit / miss.
- Every thinker prefill shows `#cached-token: 0`.
- The talker prefill starts after thinker `stage_complete`.
- Event frames match client samples.
- The `T32` request has no talker events.

**Container B′: baseline + diagnosis ($2.90).** One selected server session with the full workload. Also run one extra unmeasured `S32` with the torch profiler on (`SGLANG_TORCH_PROFILER_DIR`, `/start_profile`; verify the route in Step 0′). Read GPU kernels and process overlap from the trace; event gaps alone do not establish GPU idleness. Output: a TTFA timeline per request, talker ms/frame, the `Sfull` − `S32` difference, and the trace reading. **Decision:** choose one or two levers from the table above using this evidence, set the win threshold from the measured spread, and write the agent expectation.

**Container C′ (and optional D′): A/B ($4.10 each).** One container runs three server sessions: baseline, lever, baseline (ABA), each with the full workload. Restarting within one container keeps hardware and image fixed, and the second baseline exposes drift. Raise `timeout` to 2,400 s for three sessions (startup with a warm JIT cache is measured in A′; reduce to two sessions if a startup exceeds 6 min). D′ tests the second lever, or L1 + L3 if L1 slowed the thinker.

**Explain.** For a win, show which interval shrank. For no win, show where the time moved (for example, L1 removes the wait but the thinker slows). Write FINDINGS.md with the customer recommendation.

**Upstream follow-up decision.** After findings are written, search the pinned source, current upstream source, issues, and PRs for the observed behavior. Open an issue or propose a PR only if there is a reproducible defect or missing capability, a minimal example, a clear owner in the request path, a targeted regression test, and evidence that the proposed change improves the relevant outcome without breaking correctness. A documented config recommendation or a negative result completes this experiment too. [PR #454](https://github.com/sgl-project/sglang-omni/pull/454) is a study example for this process, not an FP8 task to repeat.

## Budget

Rates are $0.001344/s for an H100 with 8 cores and 64 GiB. Worst case is (600 s startup + timeout + 60 s idle) × that rate.

| Container | Timeout | Worst case |
| --- | ---: | ---: |
| Step 0′ (CPU) | — | ≈ $0.05 |
| A′ smoke gate (+ MPS probe) | 1,800 s | $3.31 |
| B′ baseline + diagnosis | 1,500 s | $2.90 |
| C′ ABA, lever 1 | 2,400 s | $4.11 |
| D′ ABA, lever 2 (optional) | 2,400 s | $4.11 |
| **Through C′ / with D′** | | **$10.37 / $14.48** |

The last recorded total is $2.693673 of the approved $30. A–C actually cost $0.30–0.35 each. Before each launch: `modal app list`, reconcile `modal billing report --json`, and write the cost-log line. Afterwards record actual spend and confirm the app has stopped. Stop and ask before anything beyond D′.

## Agent expectation (written 2026-10-27, before any speech run; not the user's prediction)

**Baseline (conditional on a viable, selected speech configuration; text-only BF16 timings may not transfer):**
- `S32` TTFA about 0.4–0.8 s: the thinker finishes at about 215 ms (experiment 1), then talker prefill, about 10 talker steps, and the first code2wav window.
- `Sfull` TTFA about 230 ms later than `S32` (48 extra tokens × 4.8 ms).
- `T32` TTFT within about 10 ms of experiment 1's 67.8 ms.
- RTF well below 1.

**Levers:**
- L1 is the biggest lever. It removes most of the reply-length dependence and saves more than 100 ms at `S32` and more than 300 ms at `Sfull`, at a cost of a few percent in thinker ms/token.
- L2 saves roughly 8 × talker ms/frame.
- L3 either cannot start on Modal or makes little difference for a single request.

## Stop rules and limits

Stop if colocated startup fails or runs out of memory, audio cannot be decoded, events cannot be correlated, or frames do not match samples. Keep client-only TTFA/RTF as a qualified result when events fail but audio is valid. A lever that changes the text output or causes stalls is reported as a failed lever, not tuned until it passes. Speech quality (WER, speaker similarity), concurrency, other GPUs, FP8, and engine code changes are out of scope. If the diagnosis points to a fix that needs a code change, write it up as a candidate upstream issue/PR with its evidence and stop there.
