# Experiment 3 findings: speech-output latency and the talker partial-start lever

## What ran

One H100 80GB HBM3 on Modal ran the SGLang-Omni `8200f499` colocated speech server (`qwen3_omni_colocated_h100_bf16.yaml`, BF16 Qwen3-Omni at `26291f79`) with the thinker radix cache off. The input was the 30 s LibriVox clip (409 thinker tokens) with the transcription instruction. Thinker decoding was greedy, the talker used upstream sampling with `seed: 1234`, and every request was a single sequential streaming request from localhost. These are client-observed latencies plus in-engine request events. They are not offline throughput, concurrency, or speech quality.

| Container | Run | Purpose | Actual cost |
| --- | --- | --- | ---: |
| Step 0′ | `20261027T235454Z-9af2b068-speech-cpu` | CPU config and lever resolution | $0.006475 |
| A′ | `20261027T235833Z-882c4881-speech-smoke` | Smoke gate (passed) + MPS start probe | $0.727233 |
| B′ | `20261028T000826Z-0fa799c7-speech-measure` | Baseline + one torch-traced request | $0.502370 |
| C′ | `20261028T001612Z-2ec9853e-speech-ab-L1` | Base → L1 → base server sessions in one container | $0.707274 |

Total for experiment 3 is $1.943352. The account total after C′ was $4.647829 of the approved $30, and every app is stopped. Each server session ran 3 unmeasured warmups and then 7 measured requests: 3 × `S32` (32-token reply), 2 × `Sfull` (80-token reply), and 2 × `T32` (text only). All 28 measured requests in B′ and C′ are valid. Each had an encoder-cache miss, the expected completion count, and frames from the events equal to the client's decoded samples.

## Results (ms, median [range])

| Session | `S32` TTFA | `Sfull` TTFA | `S32` TTFT | `S32` completion | Thinker ms/tok (`S32`) | Talker ms/frame | RTF | Min playback slack |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B′ base | 343 [333–344] | 656 [651–661] | 72.2 | 868 | 6.61 | 5.55 | 0.069 | 718 |
| C′ base1 | 332 [328–334] | 630 [628–631] | 70.2 | 841 | 6.35 | 5.40 | 0.067 | 720 |
| **C′ L1** | **226 [225–231]** | **223 [221–225]** | 67.8 | 793 | 8.95 | 6.45 | 0.074 | 671 |
| C′ base2 | 346 [342–347] | 672 [672–673] | 70.9 | 875 | 6.64 | 5.60 | 0.070 | 718 |

`T32` TTFT was 66.8–71.0 ms in every session. Reproduce the table with `speech_report.py <run>` and `3_speech_output/table.py`.

**Guardrails for L1, all held:**
- **Same output:** every measured request's thinker text and its decoded-audio SHA-256 are byte-identical across base1, L1, and base2.
- **No stalls:** minimum simulated playback slack is ≥ 664 ms and total stall time is 0. RTF stays at 0.074, about 13× faster than real time.
- **Memory:** 77.5 GB after readiness, the same as base.

## Analysis

**Where baseline TTFA goes (`S32_v01` in B′, 333 ms).** Dispatch to thinker first token takes 69 ms. The thinker then decodes the other 31 tokens in **193 ms (58%)**. Hand-off and talker prefill take 15 ms. The first 10 talker frames take 47 ms (14%), and code2wav plus delivery take 8 ms. The colocated config sets `enable_partial_start=False`, so the talker waits for the thinker's whole reply. That is why `Sfull` (48 more tokens) adds about 300 ms before the first audio.

**L1 (`--talker_ar.factory.enable_partial_start true`) clears the win threshold by a wide margin.** The threshold was 30 ms off median `S32` TTFA. **`S32` TTFA falls by 106–120 ms (−32 to −35%)** against the two surrounding baselines. **`Sfull` TTFA falls by 407–449 ms (−65 to −67%)**. TTFA is now independent of reply length: about 225 ms for both 32 and 80 tokens. The two baselines bracket L1 and differ from each other by only 14 ms (`S32`), so drift does not explain the effect. The agent expectation was 180–230 ms for `S32` and 200–260 ms for `Sfull`. Both medians fell inside it.

**What changed on the timeline (L1, `S32_v01`).** The talker is queued at 100 ms, about 35 ms (about 5 thinker tokens) after the thinker's first token. Its first frame arrives at 120 ms and the first window is ready at 217 ms. The thinker finishes at 342 ms, after the first audio.

**The cost is GPU sharing.** While the talker runs, thinker decode slows from about 6.4 to about 9.0 ms/token (+41%) and talker frames from about 5.5 to about 6.5 ms (+18%). The first 10 talker frames now take about 97 ms instead of 47. That is now the largest single interval before the first audio. **The full text arrives about 80 ms later** for `S32`, because the thinker finishes at 342 instead of about 262 ms. First text (TTFT) is unchanged. Total request completion is still 48–82 ms shorter for `S32` and 150–250 ms shorter for `Sfull` than in the two baselines, because the stages overlap. Text-only requests (`T32`) are unaffected (5.88–6.10 ms/token), because they never start the talker.

**Thinker decode is slower on the speech server even without the talker (side finding, not diagnosed).** `T32` decodes at 5.9–6.5 ms/token here, against 4.2–4.8 ms/token in experiments 1–2 on the text-only server. The B′ torch trace shows about 4.2 ms of thinker kernels per step followed by about 3.6 ms with no thinker kernels (traced: 7.8 ms per step). So the slowdown is idle time between steps, not slower kernels. The trace covers one process, and CPU stacks were off. The cause is unknown.

## Recommendation (for this workload)

For a single-GPU Qwen3-Omni speech deployment at SGLang-Omni `8200f499` that serves one request at a time, enable talker partial start: `--talker_ar.factory.enable_partial_start true` on top of the colocated H100 BF16 config. In this transcription-to-speech probe it cut time to first audio from about 340 to about 225 ms for short replies, and from about 650 to about 225 ms for 80-token replies. Text and audio were identical, and playback stayed about 13× faster than real time. The trade-off is that the full text finishes about 80 ms later, and both models run slower per step while they overlap. Choose it when time to first audio is what users notice. If a client needs the full text before speaking, it does not help.

## Limits

- Single sequential requests, one input clip, n = 3 / 2 per condition, and one container per session order. Concurrency (where shared-GPU contention grows) is untested. The upstream concurrency-8 table may reverse the trade-off.
- Transcription-to-speech, not a spoken-question assistant workload. Reply lengths are 32 and 80 tokens.
- Events give intervals; they are not GPU time. The trace covers one profiled request and one process at a time, with profiler overhead (thinker 7.97 against about 6.3 ms/token).
- The upstream colocated profile disables partial start, and its reason is not recorded in the pinned source. Correctness held here for 28 requests; that does not prove it under load or for other inputs.
- The data-contract gaps from experiments 1–2 also apply here: no separate manifest, resolved config, or per-run cost file, and no `validate.py` record yet for the speech folders.

## Next

1. Optional D′ (needs a decision; worst case $4.11): **L1 + L3 (MPS)**. Under L1 the thinker and talker overlap and both slow down, and the first-window interval doubled to about 97 ms. The plan's rule says to test L1 + L3 when L1 slows the thinker. A′ showed that MPS starts on Modal. Alternative: L1 + L2, which needs L2 tested alone first under the plan's rule.
2. Extend `validate.py` to `results/omni-pipeline-probe/speech/` (checksums + validation record), and archive the speech results to the `omni-probe-backup` Volume.
3. Possible upstream follow-up, only with more evidence: (a) the speech server's idle gap between thinker decode steps, which needs a CPU-stack trace and a text-only comparison trace; (b) whether the colocated H100 profile should enable partial start for low-concurrency serving. Search upstream issues/PRs first.
