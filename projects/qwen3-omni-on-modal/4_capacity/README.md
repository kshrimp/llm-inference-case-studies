# Experiment 4: capacity planning (systems)

**Question.** How much 8k-in / 1k-out traffic can one H100 serve before a frozen latency target fails, and what is the steady-state price per 1,000 requests at that rate? This is the sizing question a Modal customer asks before choosing GPU count and autoscaling limits.

## Step 4a: memory and closed-loop scan (added after experiment 1)

Experiment 1 found that the default BF16 KV pool holds 25,161 tokens, so only **two** 9,000-token requests fit at once, while 14.1 GiB of GPU memory is never used. Without fixing this first, the sweep below would only measure the KV limit. One container:

1. BF16, default `mem_fraction_static` (0.784): closed-loop scan at concurrency 1, 2, 4, 8 (requests beyond the pool queue). Record KV tokens, peak memory, step time, and output tok/s.
2. BF16, explicit `--thinker.engine.mem_fraction_static` 0.90 (and 0.93 if 0.90 has ≥ 4 GiB never used at peak): the same scan at 1, 2, 4, 8, 12, 16. **Gate:** no OOM, no failed request, and peak memory ≤ 95% of the GPU.
3. If experiment 2 passes B, the same for FP8 at the highest passing fraction.

Output: KV capacity, step time vs concurrency, and the closed-loop maximum throughput per condition. These set the open-loop rates below.

## Frozen pass criteria (fixed before any run; changing them requires a dated note and invalidates earlier sweeps)

| Criterion | Value |
| --- | --- |
| Workload | 50 distinct 8,000-token copy-task prompts ("Repeat this text exactly:" over windows of the pinned Gutenberg text starting at 50 distinct offsets 500 words apart, each sized with the server's token count as in experiment 1; the text has about 43,000 words after the start marker, so windows overlap, which costs nothing extra because the radix cache is off), `max_tokens` 1,000, greedy, radix cache off. The chat API has no `ignore_eos`; each text is longer than the cap, so a valid request stops on length at exactly 1,000 tokens |
| Arrival process | Open loop, Poisson, seed fixed per rate |
| Latency target | p95 TTFT ≤ 3.0 s **and** p95 time between tokens ≤ 60 ms, over completed requests |
| Duration per rate | 60 s warm-up (excluded), then 300 s measured window |
| Minimum sample | ≥ 60 completed requests in the measured window, or the rate is reported as under-sampled and not used |
| Throughput check | completed rate ≥ 95% of offered rate in the measured window |
| Queue bound | client-side in-flight requests ≤ 64; reaching it fails the rate |
| Failure rule | any error, timeout (120 s), or output shorter than 1,000 tokens fails the rate |
| Rates | Fixed fractions of the condition's closed-loop maximum throughput from step 4a: 0.25, 0.5, 0.7, 0.85, 1.0, 1.1×, stopping at the first failing rate; one bisection step between the last pass and the first fail if time allows. The rule is frozen now; the req/s values follow from 4a |

A rate passes only if every criterion holds. The reported capacity is the highest passing rate.

**Dated notes (2026-10-30, before the 4b run; the criteria above are unchanged):**
- *Rate definitions, as implemented:* the offered rate is the requests actually sent in the measured window ÷ 300 s, and the completed rate is the requests that finished inside the window ÷ 300 s. Comparing against the Poisson mean instead would fail a perfect server about 30% of the time at 0.35 req/s, purely from arrival noise.
- *Fractions not run:* BF16 0.90 at 0.25× (0.18 req/s, about 55 expected requests) and BF16 0.784 at 0.25× and 0.5× (0.08 and 0.16 req/s). Each expects fewer than the 60-request minimum in 300 s, so the frozen rule would report it as under-sampled and unused. BF16 0.784 runs 0.7× and 1.0× only (user-approved scope, 2026-10-29). The bisection step is not run: the container's 80-minute timeout does not leave room for it.
- *FP8 maximum:* the closed-loop maximum is the larger of 4a's 1,390 tok/s and new c32 and c40 levels run at the start of the FP8 session, fixed before any open-loop rate.
- *Prompts and compiles:* the 50 prompts are sized once to 8,000 ± 10 tokens with the server's count. Every session first sends each prompt once with `max_tokens` 1. That checks the counts against the sized set and triggers per-shape compiles before measurement.
- *Request deadline:* 120 s for the whole request (not per read), as frozen.
- Runner: `app.py --stage capacity-sweep`; analysis: `table_4b.py` recomputes every verdict from the raw records.

## Conditions

BF16 at its best passing memory fraction from 4a (and the default 0.784, for the customer comparison), then FP8 condition B if experiment 2 passes it. One container, server restarted between conditions.

## Recorded

Per request (client clock): send time, TTFT, every chunk arrival, completion, error. Per rate: offered rate, completed rate, p50/p95/p99 of TTFT and time between tokens, maximum in-flight. Engine running/queued counts and KV usage are polled every second where SGLang-Omni exposes them, on the container clock, and are not mixed with client timings.

## Price per 1,000 requests

This is a **price-based steady-state estimate**: $4.838 per GPU-container hour (see [`budget.py`](../budget.py)) ÷ (passing rate × 3,600) × 1,000. It assumes a container fully busy at that rate and excludes cold starts and idle time. `modal billing report --json` checks the experiment's total spend; it does not measure a per-condition hourly price.

**Validity:** the sweep is valid only if the server was not restarted within a rate window, every request record is present, and the prompts' token counts match the stored set.
