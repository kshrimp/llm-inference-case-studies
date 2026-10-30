# Omni pipeline probe (SGLang-Omni, Qwen3-Omni, one H100)

For one **audio-in, text-out** request, how much client-observed time passes before text appears, and which recorded pipeline intervals explain that delay? As audio duration grows, does the encoder or thinker become the larger contributor? The [long-context probe](../long-context-probe/README.md) motivates the question, but its Qwen3-0.6B/L4 timings are not a performance baseline for Qwen3-Omni/H100. This study measures single-request serving latency and request-level events, not offline throughput, kernel time, concurrency, or goodput.

**Status:** experiments 1–3 complete. Step 0 (CPU), container A smoke, and two independent measurement containers B and C ran; all 20 measured requests pass the strengthened request-level checks. **Results:** at the same 409-token thinker prompt, 30 s audio reaches first text in 67.8 ms vs 40.2 ms for text, almost all of the difference before the thinker ([exp 1](1_stage_breakdown/FINDINGS.md)). From 10 to 90 s of audio, TTFT rises 66 ms, 69% of it in CPU preprocessing, while the encoder adds only 7 ms ([exp 2](2_audio_length_validation/FINDINGS.md)). Unexplained: successive thinker output chunks are about 0.5 ms farther apart in the audio condition; output token sequences also differ. Total probe cost $1.142945 (CPU setup $0.170214, A $0.302128, B $0.351583, C $0.319020); all apps stopped.

**Evidence preservation:** [validate.py](validate.py) writes `SHA256SUMS` and `validation.json` into each of the 5 saved run folders: 2 setup, 1 smoke, 2 measurement. Both measurement runs are `valid` with 10/10 measured requests and unique request IDs, and the smoke gate is `valid`. `validate.py --check` re-verifies every checksum. `validation.json` also lists the data-contract (private study notes) artifacts this layout lacks (separate manifest, resolved config, workload, resources and summary files, and per-run cost), each with the reason. These runs are therefore exploratory evidence, not publishable contract-compliant runs. The git-ignored `results/omni-pipeline-probe/` (inputs, all run folders and tables, 40 MB) is archived as `omni-pipeline-probe-results-20261026.tar.gz` (SHA-256 `fdfd4736…9a46`) in the Modal Volume `omni-probe-backup`, and a download round-trip matched the hash. Restore with `uv run modal volume get omni-probe-backup /omni-pipeline-probe-results-20261026.tar.gz . && tar -xzf omni-pipeline-probe-results-20261026.tar.gz -C results`.

**Purpose:** exploratory practice with an unfamiliar serving stack, multimodal workloads, and a large model on Modal.

**Evidence limit from the review:** the B/C request records and raw events reproduce the tables, but the run layout does not yet include every artifact and checksum in the project's full serving data contract (private study notes). Treat these as exploratory measurements. Before using them in a public report, package the raw files with checksums and a validation record, state unavailable fields with reasons, and retain the original manifests unchanged.

## Read and run in this order

| Experiment | Question | Status | Start here |
| --- | --- | --- | --- |
| 0. Setup and smoke | Can the pinned image, weights, audio route, events, token accounting, and cache controls work on one H100? | Done: gate passed | [Step 0](#step-0-cpu-only-setup-and-source-checks) and [GPU gate](#container-a-gpu-smoke-and-decision) below |
| 1. Stage breakdown | At one bounded audio duration, what are client TTFT and the observed pipeline intervals? Text is a descriptive path check | Done : audio +28 ms TTFT vs text at 409 tokens, mostly preprocessing + encoder | [README](1_stage_breakdown/README.md) |
| 2. Audio-length validation | At three bounded audio durations, which intervals grow and by how much? | Done : 10→90 s TTFT +66 ms, preprocessing +45.5 ms | [README](2_audio_length_validation/README.md) |
| 3. Speech output (optional) | With text + speech output on one H100, where does time to first audio go, and which config lever lowers it without stalls or changed output? | Done through C′: the thinker's whole reply is 58% of TTFA, because the colocated config holds the talker until the thinker finishes. Talker partial start cuts TTFA 340→226 ms (32 tok) and 650→223 ms (80 tok), with identical output; the thinker is 41% slower while they overlap. Cost $1.94 | [README](3_speech_output/README.md), [FINDINGS](3_speech_output/FINDINGS.md) |

## Shared decisions

| Choice | Decision | Why |
| --- | --- | --- |
| Engine | `sglang-omni` v0.1.9, commit `8200f49934d66de591c035cab1c626c0600905fa` (Apache-2.0) | Pinned release; main moved beyond this commit (`85a4a057` at the time of the check) |
| Base image | `lmsysorg/sglang@sha256:5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e` | The digest the v0.1.9 `docker/Dockerfile` builds from. We add the omni package at the pinned commit with its dependency overrides |
| Model | `Qwen/Qwen3-Omni-30B-A3B-Instruct` at revision `26291f793822fb6be9555850f06dfe95f2d7e695`, BF16 | Official checkpoint. The repo's FP8 config points at a community quantization (`marksverdhei/Qwen3-Omni-30B-A3B-FP8`); FP8 would add a second variable and unclear provenance |
| Mode | `--text-only` for experiments 1–2 | Upstream docs: text-only mode runs the thinker pipeline on a single GPU and does not load the talker/code2wav. Speech mode needs either 2 GPUs or the colocated config (experiment 3) |
| GPU | `gpu="H100!"`, one H100 SXM 80 GB, one worker/container at a time | Plain `H100` can be upgraded to H200 by Modal, changing the benchmark hardware; check observed GPU at startup |
| Thinker shape | 48 layers, 4 KV heads, head_dim 128, 128 experts with top-8 routing (about 3B active parameters), 65,536 max positions | From the checkpoint's `config.json`. KV cache is 48 × 4 × 128 × 2 (K, V) × 2 bytes ≈ **96 KiB per token**, so 9,000 tokens ≈ 0.85 GiB |
| Workload | Fixed instruction and nested 16 kHz mono speech clips (10 / 30 / 90 s) cut from one continuous public-domain LibriVox passage, one distinct-sample variant per measured request; greedy output capped at 32 tokens in B/C | A used up to 128 tokens for compatibility; B/C used 32 so every measured request had the same output count |
| Measurement | Sequential requests from a localhost client, plus request-level event recorder | Label results **client-observed (localhost, single request)** and **in-engine events**, never as offline throughput |

The original 8,000-prompt / 1,000-output idea is a **conditional extension** after this bounded probe. It needs matched controls, accepted audio duration, memory headroom, and cost reservation. Do not assume API `usage.prompt_tokens` equals the number of positions the thinker processes for audio; verify in pinned source and smoke events before matching modalities by token count.

### What gets measured per request

| Metric | Source | Kind |
| --- | --- | --- |
| End-to-end time, time to first **nonempty text** chunk | client monotonic timestamps from dispatch to terminal and first content on streaming `/v1/chat/completions` | client-observed |
| Prompt tokens, completion tokens, `finish_reason`; actual output text | raw response `usage` and chunks; validate the relation between API prompt count and thinker positions | server-reported / client-observed |
| Preprocess, encoder, queue, thinker and hand-off intervals | only start/end event pairs actually emitted by the pinned text-only path and correlated to this request | in-engine events |
| Decode proxy, if supported | thinker first-token to thinker terminal event on one clock, divided by subsequent thinker output tokens; otherwise null with reason | derived from events |
| KV pool size, GPU memory | server startup log, `nvidia-smi` | server-reported |

An event that does not appear is stored as `null` with a reason. Record event source, process, request/stage ID, and clock. Never subtract timestamps from different clocks without a documented mapping. Intervals can overlap and include queueing, CPU work, and synchronization: do not sum them into an end-to-end partition or call first streamed chunk "prefill end." Client chunk gaps are not inter-token latency. Compare one short request with recording off and on to expose profiler overhead.

### Output length and cache state

The SGLang-Omni chat API at this pin has no established `ignore_eos` field. A used `max_tokens: 128` for compatibility; B/C used `max_tokens: 32`, and every measured request returned 32 tokens with `finish_reason=length`. Record actual completion count and finish reason for any later run; a longer decode-focused workload is a separate extension.

Reset the thinker prefix cache before **each measured request**, or disable it using a source-confirmed setting. Record the method, result, and any cache counters. Check for a separate media/preprocessor cache. If the state cannot be controlled, label the results warm-sequence observations and do not attribute differences to modality or duration. Keep container startup and model-file cache separate from prefix-cache state.

Before any quoted run, follow the data contracts (private study notes): `schema_version`, source/model/image/GPU pins, exact settings and clocks, request IDs, input hashes and WAV durations, raw usage and outcomes, cache protocol, failure/cleanup, artifact checksums, and a valid/qualified/invalid verdict. Validate ID correlation, event order, finite intervals, token counts, and hashes. Show individual repeats; medians are descriptive only.

## Step 0: CPU-only setup and source checks

No GPU is launched in this step.

1. **Image build (CPU):** Modal image from the base digest above; install sglang-omni from `git+https://github.com/sgl-project/sglang-omni@8200f499…` using its `override-dependencies`, as the upstream Dockerfile does. Check `import sglang_omni` and `sgl-omni serve --help` in a CPU container.
2. **Weights Volume (CPU):** download the pinned revision into a new Volume `omni-hf-cache` (70.5 GB, done once). Record file count and total bytes.
3. **Inputs (local, done):** fixed text and deterministic 16 kHz mono WAVs. Use one short intelligible speech fixture in the smoke; build duration variants from the same continuous speech content so silence detection or content changes do not explain the timing trend. Seeded noise is suitable only if the pinned path proves it is processed like speech. Record the source, SHA-256, sample count and exact duration.
4. **Source reading (local):** confirm the text-only profile route, exact emitted events and clock, request IDs, response usage, thinker input positions, cache control, and audio length limit at the pin. Fill the [stage map](1_stage_breakdown/README.md#stage-map) with file:symbol references. Missing support narrows the question before launch.

### Step 0 results

**Image and CLI check:** completed (`results/omni-pipeline-probe/setup/20261026T025017Z-e4571d72-cpu-check/run.json`). The image builds from the pinned base digest with sglang-omni checked out at `8200f499` (verified with `git rev-parse`). Installed: Python 3.12.15, torch 2.14.1+cu130, sglang 0.5.22, sglang-omni 0.1.9, transformers 5.19.0, flashinfer-python 0.7.1, flash-attn-4 4.0.0b35. `sgl-omni serve --help` exits 0. `sgl-omni config resolve --text-only --thinker.engine.disable_radix_cache true --show diff` prints exactly one change, `stages.thinker.engine.disable_radix_cache: None -> True`, so the cache override parses on CPU. Whether the engine honours it is checked in A. That first run reported `--allowed-local-media-path` as missing only because typer's help box cut off the flag name; `cpu_check` now sets a wide terminal.

**Weights:** completed (`results/omni-pipeline-probe/setup/20261026T025224Z-81e915bb-download/run.json`). Volume `omni-hf-cache` holds the pinned revision at `/weights/Qwen3-Omni-30B-A3B-Instruct`: 25 files, 70,530,595,574 bytes, every size matching the Hub. All 15 LFS files match their Hub SHA-256; the 10 small non-LFS files are checked by size only. The download took 6.5 minutes. Both Step 0 apps are stopped. Actual cost was $0.168302 against a $0.54 reservation.

**Speech inputs:** built locally by [make_inputs.py](make_inputs.py); every hash is in [inputs_manifest.json](inputs_manifest.json), and the WAVs are in git-ignored `results/omni-pipeline-probe/inputs/wav/`.

| Field | Value |
| --- | --- |
| Recording | LibriVox [*The Art of War*](https://librivox.org/the-art-of-war-by-sun-tzu/) (Sun Tzu, tr. Lionel Giles), section 1 "Laying Plans / Waging War", solo reading by **Moira Fogarty**. Public domain in the US per LibriVox; the archive.org item states a public-domain license |
| Download URL | `https://www.archive.org/download/art_of_war_librivox/art_of_war_01-02_sun_tzu_64kb.mp3` (64 kbps MP3, 22.05 kHz mono, 506.8 s) |
| Download hash | SHA-256 `6cb289fa9a13cfc94b18e8a89159da687c60db88c2b6377311f9bc5664d5492f`, 4,055,167 bytes; MD5 and SHA-1 match archive.org's file metadata |
| Clip offsets | All clips start at **65.000 s**, inside a 63.85–65.95 s pause found by a 50 ms RMS scan, well after the LibriVox preamble. Ends at 75 / 95 / 155 s. The clips are nested: the 10 s clip is a prefix of the 30 s clip, which is a prefix of the 90 s clip |
| Processing | soundfile 0.13.1 / libsndfile 1.2.2 decode → scipy `resample_poly` 320/441 to 16 kHz → round to int16 PCM WAV. Two reruns gave a byte-identical manifest |
| Variants | 8 per duration (`v00`–`v07`). Variant k ≥ 1 adds +1 LSB at sample 8000 + k (0.5 s in, about −90 dBFS). All 24 `decoded_int16_sha256` values are distinct, checked by reading each file back |

The clip ends are fixed sample counts, so a clip can end mid-word. Container A records the transcript the model returns for `v00` of each duration as a content check.

**Source findings at the pin** (details and file references in the [stage map](1_stage_breakdown/README.md#stage-map)):

| Question | Answer from source | Consequence |
| --- | --- | --- |
| Event recorder on the text-only server? | Yes: `mount_profiler_routes` (`serve/launcher.py`) always adds `/start_request_profile` and `/stop_request_profile`. `event_dir` is required unless `SGLANG_TORCH_PROFILER_DIR` is set | Pass `event_dir` explicitly. Compare recording off and on in A |
| Event clock | `time.time_ns()` everywhere; one JSONL file per process | Server and pipeline stamps are comparable inside one container. The client also records `time_ns` |
| Prompt count | `usage.prompt_tokens = len(state.prompt["input_ids"])` (`models/qwen3_omni/components/streaming_detokenizer.py::prompt_token_count`), i.e. the processor's input IDs, which include the expanded audio placeholder tokens | Likely equals thinker positions. A must still confirm this for one audio request |
| Audio tokens per second | Shared audio-tower math (`models/qwen3_asr/audio_lengths.py`): 100 mel frames/s; each full 100-frame window gives 13 tokens, plus a remainder through three stride-2 downsamples. So about **13 audio tokens per second**: 10 s ≈ 130, 30 s ≈ 390, 90 s ≈ 1,170 | Source-derived, to be checked against `usage` in A. The old 8,000-token target would need about 10 minutes of audio |
| **Audio encoder output cache** | The audio and image encoders keep a `StageOutputCache` (64 entries, 4 GiB, CPU). Its key hashes **every decoded audio sample** plus the target sample rate (`preprocessing/audio.py::compute_audio_cache_key`). It has no off switch | A repeated identical WAV skips the encoder. A changed header or file name is not enough. **Each measured request needs a distinct decoded sample sequence** with the same duration and nearly identical content (here, ±1 LSB at one sample). With `SGLANG_OMNI_TRACE_ENCODER_CACHE=1` the server logs `encoder_cache stage=audio_encoder action=miss\|hit req=<id> key=<short>` at INFO. **A measured audio request is usable only if its log shows `action=miss` and no `hit`** |
| Thinker prefix (radix) cache | On by default; there is no flush route on the server. `--thinker.engine.disable_radix_cache true` resolves on CPU | Disable it for measured runs. Confirm in the server log that it is off |
| Audio length limit (chat path) | `load_audio_path` decodes without `max_seconds`. The preprocessor rejects prompts longer than `max_seq_len` (32,768 in the default config) | 90 s is far below the limit. Bare local paths are allowed, but pinning `--allowed-local-media-path /inputs` is safer |
| Forced output length | Chat `SamplingParams` has no `ignore_eos` | A capped at 128; B/C froze 32 after smoke and verified every measured completion count |

## Container A: GPU smoke and decision

Start one server; record the observed GPU, memory headroom and readiness. Send one short text and one short audio request, verify that audio reaches its encoder, and compare the API prompt count with thinker positions. Inspect raw events for one request, confirm that the thinker radix cache is off in the server log, send the same WAV twice and then a different variant (expect miss, hit, miss in the `encoder_cache` trace lines, which proves the trace and the variant rule work), and compare one short identical request with event recording off/on. Try the planned longest duration only after the short cases pass. Freeze the three durations and output cap for B. If event boundaries or cache state cannot be established, report client-latency-only pilot results or stop the stage comparison.

### Runbook (written, run )

`app.py --stage smoke` starts one `H100!` container. In it, the server runs `sgl-omni serve` in text-only mode on 127.0.0.1, with `--thinker.engine.disable_radix_cache true`, `--allowed-local-media-path /inputs` and `SGLANG_OMNI_TRACE_ENCODER_CACHE=1`. The WAVs and manifest are mounted read-only, and the container first checks every WAV's decoded-sample SHA-256 against the manifest. JIT caches go to the Volume `omni-jit-cache`, so A's startup is a cold-cache startup and B's is not. Startup time is recorded and is never a request measurement. Streaming requests run one at a time, greedy, `max_tokens: 128`; the audio prompt is "Transcribe the speech in this audio exactly."

| Step | Input | Request events | Expected `encoder_cache` |
| --- | --- | --- | --- |
| `text_short` | one-sentence text prompt | on | — |
| `audio010_v01_first` | 10 s, v01 | on | miss |
| `audio010_v01_repeat` | 10 s, v01 again | on | **hit** (proves the trace works) |
| `audio010_v02` | 10 s, v02 | on | miss (proves the variant rule works) |
| `audio030_v01` | 30 s, v01 | on | miss |
| `audio090_v01` | 90 s, v01; runs only if every earlier step completed | on | miss |
| `overhead_profile_off` | 10 s, v03 | **off** | miss |
| `overhead_profile_on` | 10 s, v04 | on | miss |

Server readiness has a 780 s deadline, each request 180 s, and the function a hard 1,200 s timeout. The server process group is killed in `finally`, and its log and event JSONL are returned on success and on handled failure. If Modal's hard timeout fires, only the streamed progress lines in the local output survive.

```bash
cd /path/to/llm-inference-case-studies
uv run python -c 'import runpy; m=runpy.run_path("projects/omni-pipeline-probe/app.py"); m["main"](stage="smoke", dry_run=True)' # no GPU; prints limits, $2.50, server command
uv run modal app list # nothing from omni-pipeline-probe running
uv run modal billing report --start "$START_DATE" --end "$END_DATE" --json # reconcile; then write the cost-log line
uv run modal run projects/omni-pipeline-probe/app.py::main --stage smoke
uv run python projects/omni-pipeline-probe/smoke_report.py results/omni-pipeline-probe/smoke/<run_id>
```

Artifacts in `results/omni-pipeline-probe/smoke/<run_id>/`: `run.json` (pins, limits, GPU, input check, readiness, notes), `requests.jsonl` (one record per step: client wall/monotonic times, TTFT, completion time, raw `usage`, finish reason, text), `events.jsonl` (raw recorder rows plus source file), `server.log`, and `checks.json` from [smoke_report.py](smoke_report.py). The report marks each check true, false, or null with a reason. **Container B may be reserved only if `gate_for_container_B.pass` is true**, meaning all of: inputs match; server ready; every step completed; audio events show `modality=audio`; the cache pattern is exactly miss / hit / miss / miss…; the radix cache is off in the server log; events correlate under the client `request_id`; event order is consistent. The report also prints the prompt-count differences (30 s − 10 s against the formula's 260 tokens, 90 s − 10 s against 1,040), the engine `#new-token` values next to the API prompt counts, GPU memory after readiness and at peak, the single profile-off/on pair, and every output text for a content check. `smoke_report.py` was tested only on synthetic run folders (a passing case and an empty-log case); its regular expressions for the server-log lines are guesses until A's real log exists.

### Container A results

Run `20261026T035505Z-10b5b3db-smoke` (artifacts in `results/omni-pipeline-probe/smoke/<run_id>/`; regenerate the checks with `smoke_report.py`). Observed GPU: NVIDIA H100 80GB HBM3, driver 610.57.04. The server became healthy 208 s after launch, with a cold JIT cache, including upstream warmup. All 8 steps completed and the server exited cleanly (code 0). The app is stopped; actual cost was $0.302128 against the $2.50 reservation. **These are compatibility results from single requests, not benchmark numbers.**

**Gate for container B: pass.** Every required check is true:

| Check | Evidence |
| --- | --- |
| Inputs match the manifest | Decoded-sample SHA-256 of every WAV used, checked inside the container |
| Audio reaches its encoder | `encoder_start` with `modality=audio` for every audio request |
| Encoder cache rule | `encoder_cache` trace: v01 `miss,store` → v01 again `hit` → v02 `miss,store`; every other audio request `miss,store`. The trace's `req=` is the client `request_id` |
| Thinker radix cache off | All 9 thinker prefills log `#cached-token: 0`, including the identical repeated request, which would have matched almost all of its prompt with the cache on |
| Event correlation | Every profiled request has events under its client `request_id`; no stray IDs |
| Event order | The (stage, event) sequence is in time order for every request |

**Token accounting:** `usage.prompt_tokens` was 149 / 409 / 1,189 for the 10 / 30 / 90 s clips. The engine's `#new-token` values match exactly, so API prompt count = thinker positions for these requests. The differences, 409 − 149 = 260 and 1,189 − 149 = 1,040, equal the source formula's 13 tokens per second exactly. The text prompt had 15 tokens.

**Memory:** 66.4 GB used after readiness, 67.6 GB peak sampled (of 81.6 GB). The KV pool holds 26,633 tokens (K 1.22 GB + V 1.22 GB), which is 96 KiB per token as estimated. A 1,189-token request uses 4% of the pool.

**Process layout seen in the events:** two JSONL files, the coordinator (server process) and one pipeline process that hosts preprocessing, audio encoder, mm_aggregate, thinker and decode. Preprocessing hands the audio features to the encoder (`local_object`) and the text payload to mm_aggregate (`shm`) in parallel; the aggregate waits for the encoder.

**Smoke observations (descriptive only):**

| Step | Client TTFT (ms) | Preprocess | Audio encoder | Admission → thinker queue | Thinker first forward | Thinker continuation | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `audio010_v01_first` | 111 | 13.8 | **55.6** | 73.5 | 24.8 | 161.9 | 34 |
| `audio010_v01_repeat` (cache hit) | 51 | 8.4 | 0.4 | 14.4 | 24.5 | 159.1 | 34 |
| `audio010_v02` | 56 | 9.6 | 7.4 | 20.1 | 25.1 | 157.1 | 34 |
| `audio030_v01` | 94 | 26.9 | 23.8 | 54.2 | 27.9 | 374.0 | 80 |
| `audio090_v01` | 134 | 57.8 | 24.6 | 86.7 | 35.7 | 609.5 | 128 (length) |

All event columns are in ms. Four points matter for B's design:
1. **First-of-shape effects are large.** The first 10 s request's encoder took 55.6 ms; the next miss took 7.4 ms. The 30 s and 90 s requests were each the first of their duration, so their encoder times may include a one-time cost. **B must warm every duration** with an unmeasured variant before measured requests.
2. At 90 s, CPU preprocessing (57.8 ms) is the largest span before the thinker, ahead of the encoder (24.6) and the thinker's first forward (35.7). That is one observation, not a result.
3. Under the transcription instruction, output length grows with the audio (34 / 80 / 128 tokens), so completion times are not comparable across durations. Thinker continuation is about 4.8 ms per output token in every case.
4. Event recording off vs on (one pair, `v03` vs `v04`): TTFT 55.3 vs 59.0 ms, completion 213 vs 215 ms. Too few to estimate overhead, but no large effect.

**Report fixes after seeing the data (disclosed):** the first `smoke_report.py` pass failed two checks because the report was wrong, not the run. (1) Several events (`stage_aggregate_ready`, `stage_first_stream_chunk_sent`) are emitted by more than one stage, so ordering must be keyed by (stage, event). (2) The `Init Unified Radix Cache` log line appears even when the tree is disabled, because `create_tree_cache` passes `disable=` to it. The radix check now uses behaviour: zero cached tokens on an identical repeat. Both fixes come from the pinned source and the raw rows, not from tuning to a desired outcome. The raw `events.jsonl` and `server.log` are unchanged.

### Frozen settings for B (accepted )

| Setting | Proposal | Reason |
| --- | --- | --- |
| Durations | 10 / 30 / 90 s, `aow_*` nested clips | All accepted. 149 / 409 / 1,189 thinker tokens, far from the 32,768 limit and 4% of the KV pool |
| Instruction and output | Same transcription instruction, `max_tokens: 32`; valid only with `finish_reason=length` and 32 completion tokens | The 10 s clip already produced 34 tokens, so every duration should hit the cap. Output counts then match, and completion latency becomes comparable. TTFT stays the main metric |
| Warmup | One unmeasured request per duration (variant `v07`) and one per text control, before the measured order | First-of-shape costs seen above |
| Variants | Measured requests use `v01`–`v06`, never repeated within the container; each counts only with an `encoder_cache` `miss` for its ID | Encoder cache rule |
| Text control (exp 1) | A fixed text prompt with 409 ± 5 thinker tokens, the same count as `audio030`; cap and validity as above | Equal thinker positions are measurable, so the text path is matched on prompt length |
| Event recording | On for all measured requests | One pair showed no large overhead; overhead stays a stated limitation |
| Radix cache | Off (`--thinker.engine.disable_radix_cache true`), verified by `#cached-token: 0` in the log for every measured request | As in A |

On the user accepted these settings, pre-approved any reasonable change, raised the Modal budget to **$30 total**, and dropped the PREDICTION.md files: no user prediction is required for this probe. One detail was settled while writing the runner: the text control is the **written form of the same passage** (Project Gutenberg eBook #132, Giles translation, from "5, 6. _The Moral Law_ causes", recorded with its SHA-256 in the manifest), prefixed with "Repeat this text exactly:". Its word count is chosen inside the container by a binary search with unmeasured 1-token probes, so the server's own `prompt_tokens` lands closest to 409. The probe counts are saved in `run.json`.

**Agent expectation (written while B ran, before any B output was read; not the user's prediction).** At 30 s, client TTFT should be roughly 70–100 ms, made of preprocessing (~25 ms), the encoder (~8–25 ms) and the thinker's first forward (~28 ms) plus about 10 ms of HTTP and hand-offs. The text control at the same 409 tokens should reach first text 30–50 ms sooner, with a similar thinker first forward. Decode ms/token (~4.8) should not differ between them. Across 10→90 s, preprocessing should grow roughly linearly and contribute the largest part of the TTFT change; the warmed encoder should grow more weakly. If the warmed 90 s encoder time is much larger than A's 24.6 ms, A's value was not a first-of-shape artifact.

## Budget

Prices from [modal.com/pricing](https://modal.com/pricing), read : H100 $0.001097/s, CPU $0.0000131/core/s, memory $0.00000222/GiB/s. With 8 CPU cores and 64 GiB memory, one H100 container costs about **$0.081 per minute**. CPU image/download work and Volume storage are separate budget items; a region or non-preemptible premium would change the estimate.

| Container | Purpose | Worst case (startup 10 + run 20 + idle 1 min) |
| --- | --- | ---: |
| A | Smoke: server/audio route, event/usage/cache validation, duration choice | $2.50 |
| B | Experiment 1, then experiment 2 only if its gates pass, in one server session | $2.50 |
| C (optional) | Independent repeat or retry after B's evidence is reviewed | $2.50 |
| | **A + B** / **A + B + C** | **$5.00** / **$7.50** |

Spend after Step 0 was $1.719030, against what was then a $10 approval. On the user raised the approval to $30 total, which allowed the optional C. Actual costs: A $0.302128, B $0.351583, C $0.319020. Account total after C: $2.693673 (`--json`), leaving $27.31.

The per-container figure comes from `app.py::worst_case_usd` with `startup_timeout=600`, `timeout=1200`, `scaledown_window=60`, `max_containers=1`, `retries=0`, 8 CPU cores and 64 GiB. The timeout was raised from 900 s to 1,200 s because the server's load and warmup (upstream warmup default 600 s) run inside the function; with 900 s the requests would have almost no time left. Modal timeouts can slightly exceed their nominal boundary ([Modal timeout behavior](https://modal.com/docs/guide/timeouts)).

Before every paid launch: `uv run modal app list`, reconcile `uv run modal billing report --json`, and write the reservation in the study-plan cost log (private study notes). Afterwards, record actual spend or null with reason, save partial artifacts on failure, and confirm the app has stopped.

## Planned layout

```text
projects/omni-pipeline-probe/
 README.md this index
 app.py shared bounded Modal runner: cpu-check, download, smoke (A), measure (B/C)
 smoke_report.py checks a smoke run and decides the container B gate
 validate.py SHA256SUMS + validation.json per run; --check re-verifies
 make_inputs.py builds the LibriVox WAV clips and variants (local)
 inputs_manifest.json source URL/narrator/hash, clip offsets, per-variant file and decoded-sample hashes
 analysis.py shared reader: measurement run → one validated row per request
 1_stage_breakdown/ README, FINDINGS, run.sh, table.py
 2_audio_length_validation/ README, FINDINGS, run.sh, table.py
 speech_report.py exp 3: per-request speech timeline + A′ gate
 3_speech_output/ README, FINDINGS, run.sh, table.py

results/omni-pipeline-probe/ git-ignored: setup/, smoke/, measure/<run_id>/ (run.json, requests.jsonl, events.jsonl, server.log), inputs/, tables/
```

## Risks

- **Image build:** the upstream image compiles UCX and llama.cpp extras. We skip extras that text-only Qwen3-Omni does not need. If import fails without them, fall back to the full Dockerfile recipe.
- **Memory headroom:** the 70.5 GB disk checkpoint is not a VRAM measurement. Record loaded GPU memory, resolved KV tokens, and peak memory in A before accepting the planned audio durations.
- **Startup time:** the server warms up with an image + audio + text request before reporting healthy (`SGLANG_OMNI_WARMUP_TIMEOUT`, default 600 s). Startup time is logged separately and is not part of any request measurement.
- **Long audio:** the old 8,000-token target may require minutes of audio or hit a model limit. The bounded duration sweep comes first; any 8k extension uses a separate gate and reservation.
