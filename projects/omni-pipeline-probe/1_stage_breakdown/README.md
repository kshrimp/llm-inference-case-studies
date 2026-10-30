# Experiment 1: one-request stage breakdown

**Status:** planned. Container B follows the [project smoke gate](../README.md#container-a-gpu-smoke-and-decision). Settings frozen ([project README](../README.md#frozen-settings-for-b-accepted-)); no user prediction (waived).

## Question

For one audio-in, text-out request on Qwen3-Omni, what is the client time to first text, and which observed preprocessing, encoder, queue, thinker, and hand-off intervals contribute? A text-only request shows the alternate path at similar *verified thinker positions*; it is not an encoder-only counterfactual.

## Conditions and controls

| Condition | Input | Output |
| --- | --- | --- |
| `text_control` | "Repeat this text exactly:" + the written Gutenberg form of the same passage, prefix sized in the container to 409 ± 5 server-reported prompt tokens (B: 262 words → 409) | Greedy, `max_tokens: 32`; valid only with 32 tokens and `finish_reason=length` |
| `audio_mid` | "Transcribe the speech in this audio exactly." + the 30 s clip (variants v01, v02; 409 prompt tokens) | Same settings |

First verify that the audio route was used and record the raw `usage` fields and actual completion count. If thinker positions are unavailable or the text control cannot be matched, keep the text run as a descriptive path check and make no equal-token prefill claim. Audio-minus-text end-to-end time is **not** an isolated encoder cost because the paths may differ elsewhere.

Each `audio_mid` request (warmup included) uses its own manifest variant (`aow_030s_v01`, `v02`, …), so no two share decoded samples. Its timings count only if the server's `encoder_cache` trace shows `action=miss` for that request ID. Exclude one short warmup per path. Reset or disable thinker prefix caching before every measured request by the verified [cache protocol](../README.md#output-length-and-cache-state). Then run `text_control`, `audio_mid`, `audio_mid`, `text_control`, sequentially in one container. Preserve individual repeats and request IDs. If reset fails, stop the paired interpretation and label the sequence exploratory. Measure one profile-off/on identical short request in A; do not call profiled client times normal serving latency when overhead is material.

## Stage map

Fill this in Step 0 from the pinned `8200f499` source, then verify against A's raw event JSONL. Each row gets a file:symbol reference, actual event fields and clock. A blank or absent event remains unsupported; the example names below are **candidates**, not promised measurements.

Source columns filled from the pinned source (paths relative to the sglang-omni repo). The smoke-evidence column stays empty until container A. Text-only topology (`sglang_omni/models/qwen3_omni/config.py::text_stages`): `preprocessing → {image_encoder, audio_encoder} → mm_aggregate → thinker → decode`, all six stages in **one process named `pipeline` on GPU 0**. The coordinator runs in the server process. `decode` is the terminal **detokenizer** stage, not thinker decoding. Every event is stamped with `time.time_ns()` (wall clock, `sglang_omni/profiler/event_recorder.py::emit`), so stamps from the two processes in one container share a clock. The client records both `time.time_ns()` and `time.monotonic_ns()` at dispatch so client times can be placed on the event timeline.

| Interval | Candidate start | Candidate end | Process / clock | Pinned source | Smoke evidence |
| --- | --- | --- | --- | --- | --- |
| Admission | `request_admission` (stage `coordinator`) | — | server / `time_ns` | `pipeline/coordinator.py` submit path | Seen, coordinator file (pid 15) |
| Preprocessing (chat template, tokenize, audio feature extraction) | `preprocess_start` | `preprocess_end` | pipeline / `time_ns` | `models/qwen3_omni/components/preprocessor.py::__call__` (wraps `call_impl`) | Seen for all requests; pipeline file |
| Audio encoder | `encoder_start` (metadata `modality=audio`, `batch_size`) | `encoder_end` | pipeline / `time_ns` | `models/qwen3_omni/stages.py::create_audio_encoder_executor` (`_encode`, `_encode_batch`); the end is in `finally`, so it is emitted on failure too | Seen with `modality=audio`, `batch_size=1`; runs in the pipeline process |
| Aggregate ready | — | `stage_aggregate_ready` | pipeline / `time_ns` | `pipeline/stage/runtime.py` (join of preprocessing + encoders, `merge_for_thinker`) | Also emitted by `audio_encoder` and `thinker`; use the `mm_aggregate` one |
| Thinker queue | `scheduler_queue_enter` | `scheduler_prefill_start` | pipeline / `time_ns` | `scheduling/omni_scheduler.py` (enqueue; `emit_prefill_start_for_batch`) | Seen; about 1–2 ms |
| Thinker first forward | `scheduler_prefill_start` | `scheduler_prefill_end` (after the first executed batch returns; metadata `batch_size`) | pipeline / `time_ns` | `scheduling/omni_scheduler.py::emit_prefill_end_for_batch` | Seen; one `Prefill batch` log line per request with the full prompt, so no chunking at these lengths |
| Thinker first output | `scheduler_prefill_start` | `scheduler_first_emit`, then `stage_first_stream_chunk_sent` (stage `thinker`, `to_stage=decode`) | pipeline / `time_ns` | `omni_scheduler.py::stage_stream_messages`; `runtime.py::number_stream_chunk` | Seen; `decode` also emits `stage_first_stream_chunk_sent` (to the coordinator), so key on the stage |
| Thinker continuation | thinker first stream chunk | `stage_complete` (stage `thinker`) | pipeline / `time_ns` | `runtime.py` result routing | Seen; 34 thinker stream chunks for 34 tokens |
| Terminal | — | `terminal_response` (stage `coordinator`, `from_stage`, `success`) | server / `time_ns` | `pipeline/coordinator.py::_handle_completion` | Seen, coordinator file |
| Client TTFT | client dispatch | first nonempty text chunk | client monotonic (+ `time_ns` mapping) | client record | Client wall-clock dispatch → coordinator admission was about 7 ms (11.7 ms for the first request) |

Checked in A (`20261026T035505Z-10b5b3db-smoke`):
- The client's `request_id` reaches every event, including the scheduler's and the `encoder_cache` trace line, so correlation needs no mapping.
- The first thinker stream chunk carries one token: the thinker sent 34 chunks for 34 completion tokens.
- `scheduler_prefill_start → scheduler_prefill_end` covers one forward of the whole prompt. Each request has one `Prefill batch` line with `#new-token` equal to `usage.prompt_tokens` (149 / 409 / 1,189), so there is no chunked prefill at these lengths.

First streamed chunk is not assumed to be a pure prefill boundary. Event intervals can overlap and do not necessarily partition client latency. Report unsupported spans as null with reasons. Do not subtract timestamps from different clocks without a documented mapping.

## Analysis and acceptance

For every request, report input hash/duration, raw prompt and completion counts, finish reason, client TTFT and completion latency, each supported event interval, and GPU memory. Show individual values and descriptive median/spread for the two repeats. Explain any actual output-count or thinker-position mismatch before interpreting latency. The main result is a timeline of supported spans and gaps for the audio request; the text path is a secondary comparison.

Accept a stage claim only when the event pair appears for both audio repeats, correlates to one request, uses a valid common clock, and passes artifact validation. Otherwise publish the client observations with the missing stage explicitly marked. A request-level event interval includes scheduling and CPU overhead, not just GPU compute. No kernel trace or concurrency claim is part of this experiment.
