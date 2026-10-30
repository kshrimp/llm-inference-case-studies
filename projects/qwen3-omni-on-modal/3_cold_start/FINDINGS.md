# Experiment 3 findings: cold start, zero to first token (2026-10-30)

## What ran

The deployed App `qwen3-omni-cold-start` (`app.py::cold_app`) ran on one `H100!` per container, with the scaledown window at 60 s. Before each run, `client.py` waited until the App had no container, so every measured request started a new container. Each run sent one streamed request: experiment 1's S prompt (1,150 tokens), 32 greedy tokens. The server config is the recommended one: FP8 (community checkpoint) with `deep_gemm` dense and `triton` MoE, `mem_fraction_static` 0.90, text-only, radix cache off. BF16 used the same flags without the FP8 backends. Conditions are as revised in the README: one lever at a time, bucket count unchanged.

**All 9 measured runs are valid.** Each returned 32 tokens, and the text was byte-identical within each model (one distinct output for FP8, one for BF16). Reproduce with `table.py`; records are in `results/qwen3-omni-on-modal/3_cold_start/`.

## Results

Client: this machine's clock, from the call to the first streamed token. Container: the container's clock, from entrypoint to first token. Phases come from the server log, on the container clock.

| Condition | Run | Client → first token | Container entry → first token | Outside the container (≤) | Spawn + imports → weight load | Weight load (GB/s) | Prefill CUDA graphs | Decode graphs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 FP8, fresh caches | 1 | 275.1 s | 262.8 s | 12.3 s | 81.6 s | 60.5 s (0.48) | 103.2 s | 4.7 s |
| | 2 | 200.8 s | 193.0 s | 7.8 s | 53.6 s | 20.9 s (1.40) | 101.5 s | 4.5 s |
| | 3 | 277.6 s | 264.9 s | 12.7 s | 70.6 s | 24.4 s (1.20) | 144.3 s | 6.7 s |
| F2 prime (not measured) | — | 297.7 s | 284.4 s | 13.2 s | 84.2 s | 24.5 s (1.19) | 148.8 s | 7.4 s |
| **F2 FP8, kernel caches on a Volume** | 1 | 199.4 s | 190.7 s | 8.8 s | 71.4 s | 26.7 s (1.09) | 68.0 s | 5.8 s |
| | 2 | 174.3 s | 167.6 s | 6.6 s | 60.1 s | 22.2 s (1.32) | 60.5 s | 6.7 s |
| | 3 | 190.4 s | 182.5 s | 7.9 s | 63.9 s | 22.5 s (1.30) | 69.9 s | 8.6 s |
| B1 BF16, fresh caches | 1 | 238.1 s | 225.6 s | 12.5 s | 88.2 s | 79.3 s (0.72) | 34.2 s | 4.6 s |
| | 2 | 165.1 s | 157.9 s | 7.2 s | 56.2 s | 55.4 s (1.03) | 28.1 s | 3.8 s |
| F3 FP8 memory snapshot | — | **failed** | — | — | — | — | — | — |

The F2 priming run wrote 209 Triton files and 106 SGLang (DeepGEMM) files to the Volume; each F2 run started with 316 cached files.

**F3:** Modal logged "Creating GPU memory snapshot for Function", then "Failed creating Function memory snapshot … Not all applications are supported by memory snapshots". The container then ended with "Runner has been initializing for too long: 690 seconds". Modal started a second container, which failed the same way, and the run was stopped there per the stop-at-first-failure rule. Log lines: `snapshot_attempt/app_logs.txt`. About 20 GPU-minutes for the two attempts; no restore was ever reached.

## What this shows

1. **Zero to first token takes 3–5 minutes on this stack, and neither target is met.** FP8 median 275 s with fresh caches, 190 s with caches kept; BF16 165–238 s. The ≤ 90 s target for a normal start is missed by about 2×, and the < 30 s target needed a snapshot, which failed.
2. **For the FP8 config, the largest phase is compiling, not loading.** Prefill CUDA-graph capture took 101–144 s with fresh caches: over half the container time, against 28–34 s for BF16. FP8 weights load in 21–27 s on most runs (half of BF16's bytes). So the configuration that serves fastest is also the one that starts slowest, by about 70 s of compile.
3. **Persisting kernel caches on a Volume is the lever that works: −31% median (275 → 190 s).** It halved prefill capture (to 60–70 s) and narrowed the spread (174–199 s vs 201–278 s). It costs one Volume and environment variables (`SGLANG_CACHE_DIR`, `TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `TORCHINDUCTOR_CACHE_DIR`). About 60 s of capture remains even with warm caches, against 15 s for a BF16 server whose kernels are cached (4a). **Unexplained:** some JIT output is not in these four directories.
4. **About 55–90 s goes to process spawn and imports before weights load**, in every condition. Experiment 1 measured about 39 s for this phase with a warm Triton cache on a Volume. Part of the difference is compiling the audio-encoder CUDA graphs and other JIT work with an empty cache. This phase is the next target after capture.
5. **Weight loading varies 3× between containers for the same files** (0.48–1.40 GB/s FP8; 0.72–1.03 GB/s BF16), as in experiment 1. This is the main source of run-to-run spread for BF16.
6. **Time outside the container is small:** at most 6.6–13.2 s per run (scheduling, image start, Python import of the app, network). Larger values go with the slower runs.
7. **GPU memory snapshots do not work for this server as built.** SGLang-Omni starts several GPU processes linked by CUDA IPC, and offers no endpoint to release GPU memory before a snapshot. Modal documents both as known limits.

## Limits

- Two or three cold containers per condition; the medians are indicative, not tight.
- Short request (1,150-token prompt, 32 tokens). Cold-start time is almost entirely server startup, so a longer request would add only its own TTFT.
- FP8 with Triton dense GEMMs was not run as a cold-start condition. 2c showed a 60 s prefill capture for it when its tuned kernels were new, which is no better than DeepGEMM with caches kept.
- Client on this WSL machine over the public Modal API; another network adds its own latency to the "outside" column.
- Costs: 10 cold containers (one priming) plus two failed snapshot containers. Billing for 2026-10-30 not yet posted at writing; see the study-plan cost log.
