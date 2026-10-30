# Experiment 4 findings

## Step 4a: memory fraction and closed-loop concurrency scan (2026-10-29)

### What ran

Run `20261029T012811Z-3355f112-capacity-scan` (`app.py --stage capacity-scan`): one `H100!` (80 GB HBM3), SGLang-Omni `305273a6`, text-only servers with the radix cache off, four sessions in sequence:

- **A_0.784:** BF16 at the automatic memory fraction.
- **A_0.90:** BF16 at an explicit `--thinker.engine.mem_fraction_static 0.90`.
- **FP8c_0.90:** the community FP8 checkpoint with experiment 2's combined backends (`deep_gemm` dense + `triton` MoE) at 0.90.
- **A_0.93:** BF16 at 0.93. It ran because 0.90 left 5.9 GiB unused at peak; the plan's condition was ≥ 4 GiB.

Each concurrency level is one closed-loop burst: n streamed 8k/1k requests sent together. The prompts are copy tasks over 32 windows of the pinned Gutenberg text, starting 500 words apart, each 6,057 words (7,915–8,000 prompt tokens). Each request stops on length at 1,000 tokens. GPU memory was sampled every 0.5 s.

**All 98 measured requests are valid.** Billed **$1.19217439**; account total $10.13902963 of $50; app stopped. Reproduce every number with `table.py`.

### Results

| Session | KV pool (tokens) | Peak memory | Gate | Best closed-loop throughput | At concurrency | Time between tokens there (p50 / worst p95) | TTFT there (p50 / max) | Estimated $ per 1M output tokens |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- | ---: |
| A_0.784 (BF16, automatic) | 25,161 | 81.5% | pass | 318 tok/s | 4–8 (KV-limited at 2) | 5.9 / 7.6 ms | 12.8 / 19.7 s at c8 | 4.22 |
| **A_0.90 (BF16)** | 121,005 | 92.6% | pass | **730 tok/s** | 12 | 14.6 / 15.7 ms | 1.14 / 1.89 s | **1.84** |
| A_0.93 (BF16) | 145,792 | **95.4%** | **fail** (> 95%) | 871 tok/s | 16 | 15.9 / 17.2 ms | 1.45 / 2.51 s | 1.54 |
| **FP8c_0.90 (FP8 combined)** | 423,832 | 92.4% | pass | **≥ 1,390 tok/s** (still rising) | 32 (highest tested) | 18.0 / 19.7 ms | 2.57 / 5.12 s | **≤ 0.97** |

The dollar figure is a **price-based estimate**: $4.838 per GPU-container hour (`budget.py`) ÷ burst output tokens per hour. It assumes the GPU stays this busy and excludes cold starts and idle time. It is an upper bound on what an open-loop rate meeting a latency target can achieve; experiment 4b measures that.

Throughput per level (output tok/s):

| Concurrent | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_0.784 | 216 | 245 | 317 | 318 | | | | |
| A_0.90 | 217 | 321 | 455 | 636 | **730** | 650 | | |
| A_0.93 | 215 | 319 | 453 | 635 | 728 | **871** | 778 | |
| FP8c_0.90 | 234 | 382 | 562 | 805 | | 1,085 | 1,269 | **1,390** |

### What this shows

1. **The automatic memory setting caps BF16 at two concurrent 8k requests and about 318 tok/s.** At concurrency 4 and 8, the KV pool (25,161 tokens) runs two requests while the rest wait. TTFT reaches 6.5 s (c4) and 19.7 s (c8), while time between tokens stays at 5.9 ms. The scheduler also retracted (evicted and later recomputed) a request once, when it admitted a third.
2. **One flag, `mem_fraction_static 0.90`, gives BF16 2.3× the throughput (730 tok/s) with peak memory at 92.6%.** The KV pool grows 4.8× to 121,005 tokens, about 13 concurrent 9k requests. Past that (c16), requests queue again and TTFT jumps to 17 s. At 0.93 the pool fits 16 (871 tok/s), but peak memory reaches 95.4%, just over the 95% safety gate set before the run. **0.90 is the recommended BF16 setting.**
3. **FP8 with the combined backends gives at least 4.4× the default BF16 throughput (≥ 1,390 tok/s at c32), 1.9× BF16 at 0.90, and was not yet saturated.** Its KV pool (423,832 tokens) holds about 47 concurrent 9k requests. At every concurrency it is faster than BF16 at the same fraction (c8: 805 vs 636 tok/s; c16: 1,085 vs 650). Per-request speed at c32 is 18 ms between tokens (about 55 tokens/s per user). TTFT is 2.6 s median and 5.1 s max, because 32 simultaneous 8k prefills queue behind each other.
4. **Throughput per user falls as concurrency rises.** BF16 time between tokens goes from 4.4 ms (c1) to 14.6 ms (c12); FP8 from 4.1 to 18.0 ms (c32). Every level stays far below experiment 4b's 60 ms target, so for this workload **TTFT and queueing, not per-token speed, will set the latency limit.**
5. **Unexplained observation:** in the default-fraction session, admitting the second 8k request was followed by a 2.1 s gap before the first two-request decode step (server log 01:32:11.691 → 01:32:13.832). After that, decode ran at a normal 5.9 ms per step. It did not happen at 0.90 or 0.93. It is not explained; it inflates c2's decode-per-request figure (7.99 ms) but not its time between tokens (5.92 ms).

### Limits

- Closed-loop bursts of identical length, sent together, measure peak batch throughput. Production arrivals are spread out, so TTFT under a real rate can be better (less burst prefill) or worse (queueing). 4b measures open-loop rates.
- One container; each level is a single burst (n = 1–32 requests). FP8's maximum was not reached.
- The 95% peak-memory gate is a safety rule set before the run, not a measured OOM threshold; 0.93 ran without errors.
- The FP8 checkpoint is a community conversion; quality evidence is experiment 2's 60/60 needle set.

### Next: 4b open-loop sweep (rates frozen by the README rule)

The README rule sets the open-loop rates as fractions (0.25, 0.5, 0.7, 0.85, 1.0, 1.1×) of each condition's closed-loop maximum:
- **BF16 at 0.90:** maximum 730 tok/s, i.e. 0.73 req/s → rates 0.18, 0.36, 0.51, 0.62, 0.73, 0.80 req/s.
- **FP8 combined at 0.90:** maximum not reached. 1,390 tok/s at c32 is a lower bound. Proposed: one extra closed-loop level at c40 at the start of the 4b container, with the maximum set to the larger of c32 and c40 before any open-loop rate runs.
- **BF16 at the automatic 0.784,** for the customer comparison: maximum 318 tok/s, i.e. 0.32 req/s.

Each rate needs 60 s of warm-up plus a 300 s measured window, so 6 rates × 3 conditions is about 108 minutes, more than the 4b row's 75-minute timeout. **Proposal:** 4b runs BF16 at 0.90 and FP8 combined at full resolution, and BF16 at 0.784 at two rates only (0.5× and 1.0×). That is about 78 minutes, so the 4b ceiling rises from 87 to 92 minutes ($7.42, from $7.02). This needs your OK before launch.

## Step 4b: open-loop capacity against the frozen latency target (2026-10-30)

### What ran

Run `20261030T025640Z-e72ff679-capacity-sweep` (`app.py --stage capacity-sweep`): one `H100!`, three sessions in sequence, radix cache off. The criteria are the frozen ones in the README: p95 TTFT ≤ 3.0 s **and** p95 time between tokens ≤ 60 ms; a 60 s warm-up then a 300 s measured window; Poisson arrivals with a fixed seed per rate; at least 60 completed requests; completed rate ≥ 95% of offered; at most 64 requests in flight; no error, no request over 120 s, and every output exactly 1,000 tokens. The README's dated notes record the implementation choices made before the run.

- **Workload:** 50 copy-task prompts over windows 500 words apart, each sized to 8,000 tokens by the server (7,999–8,001). Every session re-counted all 50 and matched the sized set.
- **FP8c_0.90:** FP8 with `deep_gemm` dense + `triton` MoE. New closed-loop levels: c32 1,246 and **c40 1,436 tok/s**, so the rates used a maximum of 1,436 tok/s (4a measured 1,390 at c32).
- **A_0.90:** BF16 at memory fraction 0.90 (maximum 730 tok/s from 4a).
- **A_0.784:** BF16 at the automatic memory fraction (maximum 318 tok/s from 4a).

**All 2,504 requests are valid:** 1,947 in measured windows, 485 in warm-up periods, and 72 closed-loop. All three sessions completed; the app is stopped. The container ran about 70 minutes (≈ $5.65 by wall clock; billing rows for 2026-10-30 not yet posted). Reproduce every number with `table_4b.py`, which recomputes each verdict from the raw records and asserts it matches the stored one.

### Results (offered rate = requests actually sent in the window ÷ 300 s)

| Session | × max | Offered req/s | TTFT p50 / p95 (s) | Between tokens p50 / p95 (ms) | Max in flight | Verdict |
| --- | ---: | ---: | --- | --- | ---: | --- |
| FP8c_0.90 | 0.25 | 0.380 | 0.18 / 0.19 | 6.5 / 9.0 | 8 | pass |
| | 0.5 | 0.747 | 0.18 / 0.27 | 10.5 / 14.5 | 20 | pass |
| | 0.7 | 0.960 | 0.19 / 0.29 | 12.8 / 17.7 | 32 | pass |
| | 0.85 | 1.150 | 0.20 / 0.33 | 16.9 / 25.0 | 52 | pass |
| | **1.0** | **1.357** | 0.21 / **2.55** | 22.4 / 26.7 | 56 | **pass** |
| | 1.1 | (aborted) | 4.78 / 7.79 | 23.7 / 27.0 | 64 | fail: queue bound, p95 TTFT |
| A_0.90 | 0.5 | 0.363 | 0.19 / 0.20 | 7.7 / 14.4 | 13 | pass |
| | **0.7** | **0.490** | 0.19 / **0.31** | 10.6 / 15.1 | 18 | **pass** |
| | 0.85 | 0.627 | 0.20 / 6.95 | 13.2 / 15.9 | 22 | fail: p95 TTFT |
| A_0.784 | 0.7 | 0.210 | 0.19 / 7.88 | 5.8 / 6.2 | 6 | fail: p95 TTFT |

**Capacity per H100 (highest passing rate) and price-based cost** ($4.838 per GPU-container hour ÷ requests per hour; assumes a container kept busy at that rate, and excludes cold starts and idle time):

| Configuration | Capacity | Requests per hour | $ per 1,000 requests (est.) |
| --- | ---: | ---: | ---: |
| **FP8, DeepGEMM dense + Triton MoE, fraction 0.90** | **1.357 req/s** | **≈ 4,890** | **≈ $0.99** |
| BF16, fraction 0.90 | 0.490 req/s | ≈ 1,760 | ≈ $2.74 |
| BF16, automatic fraction 0.784 | none at ≥ 0.21 req/s | < 760 | > $6.40 |

### What this shows

1. **FP8 with the combined backends carries 2.8× the traffic of tuned BF16 within the same latency target** (1.357 vs 0.490 req/s), and the automatic BF16 setup fails even at its lowest testable rate. The ordering matches 4a's closed-loop result, and the open-loop gap is larger: BF16's smaller KV pool (121k vs 424k tokens) queues requests sooner.
2. **TTFT sets the limit, not per-token speed.** The p95 time between tokens never exceeded 27 ms (target 60). Every failure was TTFT: once requests queue for KV space or behind prefills, p95 TTFT jumps from about 0.3 s to 2.5–8 s within one rate step. For FP8 that knee is between 0.85× and 1.1× of the closed-loop maximum; for BF16 0.90, between 0.7× and 0.85×.
3. **Each new 8k prompt pauses every running stream for about 150 ms.** Gaps over 100 ms (median 142–157 ms, about one 8k prefill) make up 0.3% of FP8's token gaps at 0.25× and 3.4% at 1.0×. They raise p99 time between tokens to about 150 ms without breaking the frozen p95 target. This is the same prefill-stalls-decode effect as the long-context probe's experiment 10, now on a production engine. A smaller `chunked_prefill_size` would bound it at some TTFT cost; untested.
4. **Memory stayed under the 4a gate:** peak 92.4% (FP8) and 92.5% (BF16 0.90) of 81,559 MiB. The scheduler retracted requests at least 2 times (FP8) and 7 times (BF16 0.90) under load, counted from the last 2 MB of each server log, so these are lower bounds. Every retracted request still completed validly.

### Limits

- One container, one run per rate. Rates near a knee (FP8 1.0×: p95 TTFT 2.55 s against 3.0) could flip on a repeat. The bisection step was not run.
- Identical-length 8k/1k copy prompts; real traffic with mixed lengths would change both prefill stalls and KV pressure.
- The client ran inside the container on localhost, so network latency to a Modal web endpoint is not included.
- FP8 is a community checkpoint (quality evidence: experiment 2's 60/60 needle set).
- BF16 0.784 has no passing rate among those tested. Lower rates expect fewer than 60 requests in 300 s, so the frozen rule could not use them.
