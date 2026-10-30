# Context-length validation findings —

## Unprofiled 1,000-output runs

Six completed offline `LLM.generate` calls used the pinned mini-SGLang commit `9a91cfafe754aa85daee49998176275667eb58f2`, Qwen3-0.6B, BF16, one L4, greedy sampling with `ignore_eos=True`, and the same seeded token generator. The local records are in `results/long-context-probe/context-length-validation/<run-id>/run.json`. Each request returned exactly 1,000 tokens; the output hash matched between repeats at each input length. The runner records the expected model revision but does not independently verify the mounted snapshot hash.

| Input tokens | Repeat 1 | Repeat 2 | Mean | Repeat range |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 6.825 s | 6.233 s | 6.529 s | 0.592 s |
| 4,096 | 7.811 s | 7.905 s | 7.858 s | 0.094 s |
| 8,000 | 9.815 s | 9.804 s | 9.810 s | 0.011 s |

The means rise by 1.329 s from 1,024 to 4,096 input tokens and by another 1.952 s from 4,096 to 8,000. The 8,000-token mean is 3.281 s (50.3%) above the 1,024-token mean. Even the slowest 1,024-token run is faster than the fastest 4,096-token run, and the slowest 4,096-token run is faster than the fastest 8,000-token run. The 1,024-token repeat spread is 0.592 s, so two runs per condition support a clear directional finding, not a precise effect-size estimate or tail-latency claim.

The longer prompts extend the shorter prompts from the same seeded token generator; every record at a given length has the same prompt hash, including its later short trace. Added tokens change both context length and token content, so this remains a workload comparison rather than a content-independent causal estimate of KV length. Whole-call time combines prefill, decode, host-side work, and synchronization. It cannot assign the increase to decode attention or show that FP8 KV storage would improve time or capacity. These are offline engine timings, not client-observed serving latency.

Run IDs, in execution order:

| Input | Repeat | Run ID |
| ---: | ---: | --- |
| 1,024 | 1 | `20261023T191631Z-5fc773e4-length-1024-r1` |
| 4,096 | 1 | `20261023T191725Z-fc25b6e1-length-4096-r1` |
| 8,000 | 1 | `20261023T191812Z-2997cf4a-length-8000-r1` |
| 8,000 | 2 | `20261023T191853Z-bd9848d8-length-8000-r2` |
| 4,096 | 2 | `20261023T192051Z-cb552fe4-length-4096-r2` |
| 1,024 | 2 | `20261023T192312Z-bff29a76-length-1024-r2` |

## Matched 128-output Chrome traces

Three further runs used the same model, source pin, prompts, sampling, GPU, and engine settings, with **128 output tokens and PyTorch profiling enabled**. Each returned 128 tokens. All traces were downloaded into the matching local result folder; run IDs are:

| Input | Trace run ID | Local trace |
| ---: | --- | --- |
| 1,024 | `20261023T193306Z-cdcffeee-length-1024-trace-r1` | `results/long-context-probe/context-length-validation/20261023T193306Z-cdcffeee-length-1024-trace-r1/trace.json` |
| 4,096 | `20261023T193534Z-ad8edeff-length-4096-trace-r1` | `results/long-context-probe/context-length-validation/20261023T193534Z-ad8edeff-length-4096-trace-r1/trace.json` |
| 8,000 | `20261023T194033Z-45cfdc87-length-8000-trace-r1` | `results/long-context-probe/context-length-validation/20261023T194033Z-45cfdc87-length-8000-trace-r1/trace.json` |

Run the [decode-step analyzer](../1_profile_baseline/decode_step_trend.py) on each path with `--window 32` to reproduce the first/last medians. It finds **127 inferred decode steps**, 28 FlashInfer paged-KV decode-attention kernels per step, and 112 main cuBLAS GEMV kernels per step in every trace.

| Input | Attention sum/step, first 32 median | Attention sum/step, last 32 median | Main GEMV sum/step, first 32 median | Decode-step start interval, first 32 median |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.554 ms | 0.593 ms | 3.771 ms | 6.090 ms |
| 4,096 | 1.973 ms | 2.021 ms | 3.792 ms | 7.572 ms |
| 8,000 | 3.725 ms | 3.773 ms | 3.775 ms | 9.312 ms |

The first-step-window attention sum grows **3.171 ms** from 1,024 to 8,000 input tokens, while the interval between step starts grows **3.222 ms** and the main GEMV sum changes by only 0.004 ms. This supports a specific mechanism: longer KV context makes decode attention more expensive and is associated with longer decode steps. The earlier 8,000-input / 1,000-output trace independently shows attention increasing from 3.742 to 4.149 ms per step across generation. These are one trace per length and inferred step boundaries, so the matching increments are diagnostic, not a causal decomposition or an achievable optimization gain.

The pinned source provides the call-path link: `python/minisgl/attention/fi.py::FlashInferBackend.prepare_metadata` passes each request's `device_len` into FlashInfer metadata and selects the decode wrapper; `_initialize_metadata_once` plans with those sequence lengths and the KV dtype; `forward` stores the new K/V and calls the wrapper. The trace's `BatchDecodeWithPagedKVCacheKernel` events are therefore the FlashInfer decode-attention work on that path. The trace does not expose the scheduler's exact per-step length or hardware memory-bandwidth counters.

The GPU-event span in the 128-output traces also grows with input length:

| Input | First GPU event → first decode attention | First → last GPU event | Prefill-attention kernel sum |
| ---: | ---: | ---: | ---: |
| 1,024 | 35 ms | 819 ms | 2.5 ms |
| 4,096 | 127 ms | 1,104 ms | 38.7 ms |
| 8,000 | 303 ms | 1,500 ms | 133.3 ms |

The pre-decode interval includes all recorded GPU work before the first decode-attention kernel; it is **not** a complete prefill wall-time metric. The full GPU span is also not the profiled Python call time. Kernel duration sums can overlap, and profiler instrumentation changes timing. Nonetheless, the separate pre-decode and per-decode trends show that both stages matter. The 1,000-output timing comparison cannot be reduced to these 128-output trace values by subtraction.

Every trace run logged an 18.43 GiB KV pool with capacity for 172,544 tokens. The pool is sized from available GPU memory in pinned `python/minisgl/engine/engine.py::_determine_num_pages`; that log is **allocated capacity**, not occupied KV for one request. The 9,000-token request's BF16 logical KV payload is about 0.96 GiB. Thus this single-request experiment does not demonstrate KV capacity pressure, an FP8 speedup, or a lower Modal bill. An FP8 follow-up would need actual dtype support, quality checks, pool/occupancy measurements, and paired unprofiled runs.

## Cost and cleanup

The first `report.csv` export created 15:24 EDT contained four `long-context-length-validation` app rows totaling $0.03379862. A later Modal billing report posted six rows totaling **$0.04966881** for the six runs. This is an app-level total. An agent check of `uv run modal app list` showed all six validation apps stopped before the short-trace launches.

The three short traces were launched sequentially against a pre-run **$1.02** reservation within the $10 sprint budget. The refreshed Modal billing report shows app charges of $0.00905369, $0.00799793, and $0.00745577 respectively, totaling **$0.02450739** in posted charges. The account total in that report is $0.52293722. An agent check of `uv run modal app list` showed all three trace apps stopped with zero tasks. No FP8 treatment or additional GPU run was launched.
