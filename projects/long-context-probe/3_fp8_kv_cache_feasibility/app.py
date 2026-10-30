"""BF16 versus experimental FP8 KV storage on pinned mini-SGLang.

No upstream engine source is copied into this repository. Both images clone
mini-SGLang (MIT licensed) at one exact commit with identical pinned
dependencies; each FP8 image also applies its patch from this folder
(fp8: v1 plain cast; fp8_sat: v2 saturating cast).
"""

import collections
import datetime as dt
import gzip
import hashlib
import json
import random
import statistics
import time
import sys
import uuid
from pathlib import Path

import modal

BASE_REPO = "https://github.com/sgl-project/mini-sglang.git"
BASE_SHA = "9a91cfafe754aa85daee49998176275667eb58f2"
PATCHES = {"fp8": "fp8_kv_cache.patch", "fp8_sat": "fp8_kv_cache_v2_saturate.patch",
           # experiment 9: cache_req use-after-free regression test, without and with the fix
           "bf16_testonly": "../9_minisgl_prefix_cache_uaf/regression_test_only.patch",
           "bf16_cachefix": "../9_minisgl_prefix_cache_uaf/cache_req_uaf_fix.patch"}
TEST_IMAGES = ("bf16_testonly", "bf16_cachefix")
VARIANTS = ("bf16", *PATCHES)
HERE = Path(__file__).parent
# Modal re-imports this module in the container, where the patch lives at /opt instead.
# Worker subprocesses re-import this file as a plain script (is_local() is True there), so check files too.
PATCH_SHA256 = ({v: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                 for v, name in PATCHES.items() if (HERE / name).is_file()} if modal.is_local() else {})
# Upstream lists only lower bounds; pin both variants to the same resolved set.
PINS = ("torch==2.9.1 flashinfer-python==0.7.2.post1 sgl-kernel==0.3.21 "
        "apache-tvm-ffi==0.1.15.post0 quack-kernels==0.6.5 transformers==4.57.3 "
        "torch-c-dlpack-ext==0.1.5")
VERSION_PACKAGES = ("torch", "flashinfer-python", "sgl-kernel", "apache-tvm-ffi",
                    "quack-kernels", "transformers")
MODEL_DIR = "/root/huggingface/Qwen3-0.6B"
MODEL_REVISION_EXPECTED = "c1899de289a04d12100db370d81485cdf75e47ca"
INPUT_LENGTHS = (1024, 8000)
OUTPUT_TOKENS = 1000
TRACE_OUTPUT_TOKENS = 128
QUALITY_OUTPUT_TOKENS = 128
SEED = 80001000
QUALITY_PROMPTS = (
    "The capital of France is",
    "Explain why the sky is blue in two sentences.",
    "def fibonacci(n):\n",
    "Write a haiku about autumn leaves.",
    "List three differences between TCP and UDP.",
    "Translate to French: The weather is nice today.",
    "What is 17 multiplied by 23? Show your work.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
)


def make_image(variant: str):
    image = (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
        .apt_install("git", "build-essential")
        .env({"CUDA_HOME": "/usr/local/cuda", "TVM_FFI_CACHE_DIR": "/cache/tvm-ffi",
              "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer"})
        .run_commands(f"git clone {BASE_REPO} /opt/mini-sglang"
                      f" && cd /opt/mini-sglang && git checkout {BASE_SHA}")
    )
    if variant in PATCHES:
        image = (image.add_local_file(HERE / PATCHES[variant], "/opt/engine.patch", copy=True)
                 .run_commands("cd /opt/mini-sglang && git apply /opt/engine.patch"
                               " && git diff --stat"))
    extra = " pytest==8.3.3" if variant in TEST_IMAGES else ""
    return image.run_commands(f"cd /opt/mini-sglang && python -m pip install -e . {PINS}{extra}")


app = modal.App("long-context-fp8-kv-cache")
weights = modal.Volume.from_name("llm-bench-hf-cache")
jit_cache = modal.Volume.from_name("tiny-engine-jit-cache")

GPU_LIMITS = dict(
    gpu="L4", cpu=2, memory=8192, max_containers=1,
    startup_timeout=300, timeout=900, scaledown_window=60,
    volumes={"/root/huggingface": weights, "/cache": jit_cache},
)


def run_case(variant: str, input_length: int, capture_trace: bool) -> dict:
    import importlib.metadata
    import subprocess

    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    if not Path(MODEL_DIR, "config.json").is_file():
        raise FileNotFoundError("Model snapshot missing from llm-bench-hf-cache")
    git = lambda *a: subprocess.check_output(["git", "-C", "/opt/mini-sglang", *a], text=True)
    source = {"engine_repo": BASE_REPO, "engine_commit": git("rev-parse", "HEAD").strip(),
              "source_diff_stat": git("diff", "--stat").strip() or None,
              "patch_sha256": None}
    if variant in PATCHES:
        source["patch_file"] = PATCHES[variant]
        source["patch_sha256"] = hashlib.sha256(
            Path("/opt/engine.patch").read_bytes()).hexdigest()

    rng = random.Random(SEED)
    prompt = [rng.randint(0, 10000) for _ in range(input_length)]
    prompt_hash = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
    extra = {"kv_cache_dtype": "fp8_e4m3"} if variant != "bf16" else {}
    llm = LLM(MODEL_DIR, max_seq_len_override=9216, max_extend_tokens=16384,
              cuda_graph_max_bs=1, page_size=256, **extra)
    kv_buffer = llm.engine.kv_cache._kv_buffer
    kv_dtype = llm.engine.kv_cache.dtype
    expected = torch.float8_e4m3fn if variant != "bf16" else torch.bfloat16
    if kv_dtype != expected:
        raise RuntimeError(f"{variant} run has unexpected KV dtype: {kv_dtype}")
    element = kv_buffer.element_size()
    per_token_bytes = 2 * kv_buffer.shape[1] * kv_buffer.shape[4] * kv_buffer.shape[5] * element

    llm.generate(["Warmup"], SamplingParams(temperature=0.0, max_tokens=1))
    params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=OUTPUT_TOKENS)
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate([prompt], params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    token_ids = outputs[0]["token_ids"]

    # Quality fixtures run after timing so they cannot warm the timed call.
    quality_params = SamplingParams(temperature=0.0, ignore_eos=True,
                                    max_tokens=QUALITY_OUTPUT_TOKENS)
    quality = [{"prompt": p, "token_ids": r["token_ids"], "text": r["text"]}
               for p, r in zip(QUALITY_PROMPTS, llm.generate(list(QUALITY_PROMPTS), quality_params))]

    trace = None
    if capture_trace:
        trace_params = SamplingParams(temperature=0.0, ignore_eos=True,
                                      max_tokens=TRACE_OUTPUT_TOKENS)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                 torch.profiler.ProfilerActivity.CUDA]) as profiler:
            llm.generate([prompt], trace_params)
            torch.cuda.synchronize()
        profiler.export_chrome_trace("/tmp/trace.json")
        trace = gzip.compress(Path("/tmp/trace.json").read_bytes())

    record = {
        "variant": variant, **source,
        "kv_cache_dtype": str(kv_dtype), "kv_pool_shape": list(kv_buffer.shape),
        "kv_pool_allocated_bytes": kv_buffer.numel() * element,
        "kv_pool_capacity_tokens_excluding_dummy": (kv_buffer.shape[2] - 1) * kv_buffer.shape[3],
        "kv_bytes_per_token": per_token_bytes,
        "logical_request_kv_bytes_estimate": per_token_bytes * (input_length + OUTPUT_TOKENS),
        "input_tokens": len(prompt), "prompt_sha256": prompt_hash,
        "requested_output_tokens": OUTPUT_TOKENS, "actual_output_tokens": len(token_ids),
        "output_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
        "elapsed_seconds": elapsed, "valid_count": len(token_ids) == OUTPUT_TOKENS,
        "measurement": "offline LLM.generate whole call, unprofiled",
        "sampling": {"temperature": 0.0, "ignore_eos": True},
        "quality_fixtures": quality,
        "trace_output_tokens": TRACE_OUTPUT_TOKENS if capture_trace else None,
        "model_revision_expected_not_verified": MODEL_REVISION_EXPECTED,
        "package_versions": {p: importlib.metadata.version(p) for p in VERSION_PACKAGES},
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
    }
    return {"record": record, "trace_gz": trace}


def diagnose_case(variant: str, input_length: int) -> dict:
    """Range/NaN check of stored KV after one short generation on a zeroed pool."""
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    rng = random.Random(SEED)
    prompt = [rng.randint(0, 10000) for _ in range(input_length)]
    extra = {"kv_cache_dtype": "fp8_e4m3"} if variant != "bf16" else {}
    llm = LLM(MODEL_DIR, max_seq_len_override=9216, max_extend_tokens=16384,
              cuda_graph_max_bs=1, page_size=256, **extra)
    kv_buffer = llm.engine.kv_cache._kv_buffer
    kv_buffer.zero_()
    out = llm.generate([prompt], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=32))[0]
    torch.cuda.synchronize()
    layers = []
    for layer in range(kv_buffer.shape[1]):
        stats = {"layer": layer}
        for name, index in (("k", 0), ("v", 1)):
            max_abs, over_448, nan = 0.0, 0, 0
            for start in range(0, kv_buffer.shape[2], 64):  # chunk pages to bound memory
                chunk = kv_buffer[index, layer, start:start + 64]
                if variant != "bf16":  # e4m3fn NaN bytes are S.1111.111
                    nan += int(((chunk.view(torch.uint8) & 0x7F) == 0x7F).sum())
                values = chunk.float()
                nan += 0 if variant != "bf16" else int(values.isnan().sum())
                values = values.nan_to_num(0.0).abs()
                max_abs = max(max_abs, float(values.max()))
                over_448 += int((values > 448).sum())
            stats.update({f"{name}_max_abs": max_abs, f"{name}_over_448": over_448,
                          f"{name}_nan": nan})
        layers.append(stats)
    return {"variant": variant, "input_tokens": len(prompt), "kv_cache_dtype": str(kv_buffer.dtype),
            "first_output_token_ids": out["token_ids"], "first_output_text": out["text"],
            "layers": layers}


def bench_case(variant: str, num_page_override: int | None = None) -> dict:
    """W2 throughput workload, reproduced from pinned mini-SGLang benchmark/offline/bench.py
    (MIT; same seed, request generator, engine settings, and timing), plus decode batch sizes."""
    import importlib.metadata
    import statistics
    from random import randint, seed

    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    seed(0)
    extra = {"kv_cache_dtype": "fp8_e4m3"} if variant != "bf16" else {}
    if num_page_override is not None:  # capacity-matched control: same token pool as BF16
        extra["num_page_override"] = num_page_override
    llm = LLM(MODEL_DIR, max_seq_len_override=4096, max_extend_tokens=16384,
              cuda_graph_max_bs=256, page_size=256, **extra)
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, 1024))] for _ in range(256)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, 1024))
                       for _ in range(256)]
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))  # upstream flashinfer warmup
    decode_sizes = []
    schedule = llm.decode_manager.schedule_next_batch

    def recording_schedule():  # study instrumentation: observe, do not change, decode batches
        batch = schedule()
        if batch is not None:
            decode_sizes.append(batch.size)
        return batch

    llm.decode_manager.schedule_next_batch = recording_schedule
    t = time.time()
    outputs = llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    generated = sum(len(o["token_ids"]) for o in outputs)
    kv_buffer = llm.engine.kv_cache._kv_buffer
    return {
        "variant": variant, "num_page_override": num_page_override,
        "kv_cache_dtype": str(kv_buffer.dtype),
        "kv_pool_capacity_tokens_excluding_dummy": (kv_buffer.shape[2] - 1) * kv_buffer.shape[3],
        "requested_tokens": total_tokens, "generated_tokens": generated,
        "valid_count": generated == total_tokens, "elapsed_seconds": t,
        "throughput_tok_s": total_tokens / t,
        "requested_input_plus_output_tokens": sum(len(p) for p in prompt_token_ids) + total_tokens,
        "decode_steps": len(decode_sizes),
        "decode_batch_size_max": max(decode_sizes, default=None),
        "decode_batch_size_mean": statistics.fmean(decode_sizes) if decode_sizes else None,
        "decode_batch_size_first_10": decode_sizes[:10],
        "measurement": "offline LLM.generate throughput, upstream bench.py formula",
        "package_versions": {p: importlib.metadata.version(p) for p in VERSION_PACKAGES},
        "gpu": torch.cuda.get_device_name(0),
    }


@app.function(image=make_image("bf16"), **GPU_LIMITS)
def bench_bf16() -> dict:
    return bench_case("bf16")


@app.function(image=make_image("fp8_sat"), **GPU_LIMITS)
def bench_fp8_sat(num_page_override: int | None = None) -> dict:
    return bench_case("fp8_sat", num_page_override)


BF16_W2_PAGES = 674  # BF16 pool on the L4 in experiment 5: 172,544 tokens / page_size 256
EVAL_SEED = 20261024
EVAL_PROBLEMS = 200


def eval_problems():
    rng = random.Random(EVAL_SEED)
    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
    problems = []
    for _ in range(EVAL_PROBLEMS):
        a, b, op = rng.randint(10, 99), rng.randint(10, 99), rng.choice(list(ops))
        problems.append({"question": f"What is {a} {op} {b}? Answer with only the number.",
                         "answer": ops[op](a, b)})
    return problems


def eval_case(variant: str) -> dict:
    """Exact-answer arithmetic accuracy through the chat template (thinking off), greedy."""
    import re as regex

    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    extra = {"kv_cache_dtype": "fp8_e4m3"} if variant != "bf16" else {}
    llm = LLM(MODEL_DIR, max_seq_len_override=4096, max_extend_tokens=16384,
              cuda_graph_max_bs=256, page_size=256, **extra)
    problems = eval_problems()
    prompts = [llm.tokenizer.apply_chat_template([{"role": "user", "content": p["question"]}],
                                                 tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False) for p in problems]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=24))
    records, correct = [], 0
    for problem, out in zip(problems, outputs):
        found = regex.findall(r"-?\d+", out["text"].replace(",", ""))
        parsed = int(found[-1]) if found else None  # last integer: answers often echo "a * b = c"
        correct += parsed == problem["answer"]
        records.append({**problem, "text": out["text"], "parsed": parsed,
                        "correct": parsed == problem["answer"], "token0_count": out["token_ids"].count(0)})
    return {"variant": variant, "kv_cache_dtype": str(llm.engine.kv_cache.dtype),
            "eval_seed": EVAL_SEED, "problems": len(problems), "correct": correct,
            "accuracy": correct / len(problems),
            "outputs_with_token0": sum(r["token0_count"] > 0 for r in records),
            "records": records, "gpu": torch.cuda.get_device_name(0),
            "measurement": "greedy exact-answer arithmetic, chat template, thinking disabled"}


SWEEP_BATCHES = (1, 4, 16, 32)
SWEEP_CONTEXTS = (256, 1024, 2048, 4096)  # every cell fits the BF16 pool: 32 x 4,607 < 172,544
SWEEP_OUTPUT = 256


def sweep_case(variant: str) -> dict:
    """Experiment 6: batch x context grid; every request of a cell runs in one decode batch."""
    import statistics

    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    extra = {"kv_cache_dtype": "fp8_e4m3"} if variant != "bf16" else {}
    llm = LLM(MODEL_DIR, max_seq_len_override=4608, max_extend_tokens=16384,
              cuda_graph_max_bs=max(SWEEP_BATCHES), page_size=256, **extra)
    kv_buffer = llm.engine.kv_cache._kv_buffer
    per_token_bytes = 2 * kv_buffer.shape[1] * kv_buffer.shape[4] * kv_buffer.shape[5] * kv_buffer.element_size()
    stamps = []
    schedule = llm.decode_manager.schedule_next_batch

    def recording_schedule():  # study instrumentation: timestamps of decode batches only
        batch = schedule()
        if batch is not None:
            stamps.append((time.perf_counter(), batch.size))
        return batch

    llm.decode_manager.schedule_next_batch = recording_schedule
    rng = random.Random(SEED)

    def run_cell(batch: int, context: int) -> dict:
        prompts = [[rng.randint(0, 10000) for _ in range(context)] for _ in range(batch)]
        stamps.clear()
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = llm.generate(prompts, SamplingParams(temperature=0.0, ignore_eos=True,
                                                       max_tokens=SWEEP_OUTPUT))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        times, sizes = [t for t, _ in stamps], [b for _, b in stamps]
        gaps = [b - a for a, b in zip(times, times[1:])]
        middle = gaps[len(gaps) // 4: 3 * len(gaps) // 4] or gaps
        generated = sum(len(o["token_ids"]) for o in outputs)
        return {"batch": batch, "context": context, "elapsed_seconds": elapsed,
                "generated_tokens": generated, "valid_count": generated == batch * SWEEP_OUTPUT,
                "output_tok_s": generated / elapsed,
                "time_to_first_decode_batch_s": (times[0] - started) if times else None,
                "decode_step_ms_median_middle_half": 1000 * statistics.median(middle) if middle else None,
                "decode_steps": len(sizes), "decode_batch_max": max(sizes, default=None),
                "kv_occupancy_tokens_end": batch * (context + SWEEP_OUTPUT),
                "kv_occupancy_bytes_end": batch * (context + SWEEP_OUTPUT) * per_token_bytes}

    run_cell(1, 256)  # warmup, discarded
    cells = [run_cell(b, c) for c in SWEEP_CONTEXTS for b in SWEEP_BATCHES]
    return {"variant": variant, "kv_cache_dtype": str(kv_buffer.dtype),
            "kv_pool_capacity_tokens_excluding_dummy": (kv_buffer.shape[2] - 1) * kv_buffer.shape[3],
            "output_tokens_per_request": SWEEP_OUTPUT, "cells": cells,
            "measurement": "offline LLM.generate; step time from decode-schedule timestamps (overlap scheduler)",
            "gpu": torch.cuda.get_device_name(0)}


NEEDLE_LENGTHS = (2048, 4096, 8000)
NEEDLE_DEPTHS = (0.1, 0.5, 0.9)
NEEDLE_TRIALS = 10
NEEDLE_FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
NEEDLE_NAMES = ("Atlas", "Borealis", "Cobalt", "Dune", "Ember", "Fjord", "Garnet", "Harbor", "Iris", "Juniper")


def needle_prompts(tokenizer) -> list:
    rng = random.Random(EVAL_SEED + 1)
    filler_tokens = len(tokenizer.encode(NEEDLE_FILLER))
    items = []
    for length in NEEDLE_LENGTHS:
        for depth in NEEDLE_DEPTHS:
            for trial in range(NEEDLE_TRIALS):
                name, code = NEEDLE_NAMES[trial], rng.randint(1_000_000, 9_999_999)
                units = max(1, (length - 80) // filler_tokens)  # leave room for needle, question, template
                cut = int(units * depth)
                haystack = (NEEDLE_FILLER * cut + f"The secret code for project {name} is {code}. "
                            + NEEDLE_FILLER * (units - cut))
                question = f"What is the secret code for project {name}? Answer with only the number."
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": haystack + "\n\n" + question}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                items.append({"target_length": length, "depth": depth, "trial": trial, "name": name,
                              "code": code, "prompt": prompt,
                              "prompt_tokens": len(tokenizer.encode(prompt))})
    return items


def needle_case(variant: str, cache_type: str = "radix", lengths: tuple = NEEDLE_LENGTHS,
                num_page_override: int | None = None, detect_uaf: bool = False) -> dict:
    """Experiment 7: needle-in-a-haystack retrieval, greedy, exact code match.

    Experiment 7b ablations vary the prefix cache, the prompt subset, and the pool size.
    """
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    extra = {"kv_cache_dtype": "fp8_e4m3"} if variant.startswith("fp8") else {}
    if num_page_override is not None:
        extra["num_page_override"] = num_page_override
    llm = LLM(MODEL_DIR, max_seq_len_override=9216, max_extend_tokens=16384,
              cuda_graph_max_bs=32, page_size=256, cache_type=cache_type, **extra)
    items = [i for i in needle_prompts(llm.tokenizer) if i["target_length"] in lengths]
    uaf_events = []
    if detect_uaf:  # experiment 9 study instrumentation: observe allocations, change nothing
        allocate, page = llm.cache_manager._allocate, llm.cache_manager.page_size

        def checked_allocate(needed_pages):
            pages = allocate(needed_pages)
            fresh = set(pages.tolist())
            for req in llm.decode_manager.running_reqs:
                live = llm.cache_manager.page_table[req.table_idx, : req.device_len : page].tolist()
                if hit := fresh.intersection(live):
                    uaf_events.append({"victim_uid": req.uid, "pages": len(hit), "device_len": req.device_len})
            return pages

        llm.cache_manager._allocate = checked_allocate
    outputs = llm.generate([i["prompt"] for i in items], SamplingParams(temperature=0.0, max_tokens=16))
    records = []
    for item, out in zip(items, outputs):
        records.append({k: v for k, v in item.items() if k != "prompt"}
                       | {"text": out["text"], "correct": str(item["code"]) in out["text"],
                          "token0_count": out["token_ids"].count(0)})
    kv_buffer = llm.engine.kv_cache._kv_buffer
    return {"variant": variant, "kv_cache_dtype": str(llm.engine.kv_cache.dtype),
            "cache_type": cache_type, "lengths": list(lengths), "num_page_override": num_page_override,
            "detect_uaf": detect_uaf, "uaf_events": uaf_events if detect_uaf else None,
            "kv_pool_capacity_tokens_excluding_dummy": (kv_buffer.shape[2] - 1) * kv_buffer.shape[3],
            "correct": sum(r["correct"] for r in records), "total": len(records),
            "records": records, "gpu": torch.cuda.get_device_name(0),
            "measurement": "needle retrieval, chat template, thinking disabled, greedy, code substring match"}


@app.function(image=make_image("bf16"), **GPU_LIMITS)
def sweep_bf16() -> dict:
    return sweep_case("bf16")


@app.function(image=make_image("fp8_sat"), **GPU_LIMITS)
def sweep_fp8_sat() -> dict:
    return sweep_case("fp8_sat")


@app.function(image=make_image("bf16"), **GPU_LIMITS)
def needle_bf16(**options) -> dict:
    return needle_case("bf16", **options)


@app.function(image=make_image("fp8_sat"), **GPU_LIMITS)
def needle_fp8_sat(**options) -> dict:
    return needle_case("fp8_sat", **options)


@app.function(image=make_image("bf16_cachefix"), **GPU_LIMITS)
def needle_bf16_cachefix(**options) -> dict:
    return needle_case("bf16_cachefix", **options)


def unit_test_case(variant: str) -> dict:
    """Experiment 9: run the cache_req regression test (CPU only) on one image."""
    import subprocess

    # -o addopts="" drops the project's --cov flags (pytest-cov is not installed in the image)
    proc = subprocess.run(["python", "-m", "pytest", "-q", "-o", "addopts=", "tests/core/test_cache_req_shared_prefix.py",
                           "tests/core/test_cache_allocate.py"], cwd="/opt/mini-sglang",
                          capture_output=True, text=True)
    return {"variant": variant, "returncode": proc.returncode, "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-1500:]}


CPU_LIMITS = dict(cpu=1, memory=2048, max_containers=1, timeout=300)

# Experiment 10: prefill/decode interference. Shorts are listed first, so they prefill first and are
# decoding when the long prompts' (chunked) prefill runs; mini-SGLang schedules prefill before decode.
INTERFERENCE_CONFIGS = (
    {"name": "control", "shorts": 32, "longs": 0, "chunk": 8192},
    {"name": "1long_chunk8k", "shorts": 32, "longs": 1, "chunk": 8192},
    {"name": "4long_chunk2k", "shorts": 32, "longs": 4, "chunk": 2048},
    {"name": "4long_chunk8k", "shorts": 32, "longs": 4, "chunk": 8192},
    {"name": "4long_chunk16k", "shorts": 32, "longs": 4, "chunk": 16384},
)
SHORT_INPUT, SHORT_OUTPUT, LONG_INPUT, LONG_OUTPUT = 128, 256, 8000, 16


def interference_worker(config: dict, out_path: str) -> None:
    """One engine per config (subprocess): log every scheduled step, then derive per-request timing."""
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    llm = LLM(MODEL_DIR, max_seq_len_override=9216, max_extend_tokens=config["chunk"],
              cuda_graph_max_bs=64, page_size=256)
    rng = random.Random(SEED)
    shorts = [[rng.randint(0, 10000) for _ in range(SHORT_INPUT)] for _ in range(config["shorts"])]
    longs = [[rng.randint(0, 10000) for _ in range(LONG_INPUT)] for _ in range(config["longs"])]
    params = ([SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=SHORT_OUTPUT)] * len(shorts)
              + [SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=LONG_OUTPUT)] * len(longs))
    llm.generate(["Warmup"], SamplingParams(temperature=0.0, max_tokens=1))
    steps = []
    schedule = llm._schedule_next_batch

    def logging_schedule():  # study instrumentation: record, do not change, each scheduled batch
        forward_input = schedule()
        if forward_input is not None:
            batch = forward_input.batch
            steps.append({"t": time.perf_counter(), "prefill": batch.is_prefill,
                          "uids": [r.uid for r in batch.reqs],
                          "extend_tokens": sum(r.extend_len for r in batch.reqs)})
        return forward_input

    llm._schedule_next_batch = logging_schedule
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(shorts + longs, params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    first_uid = min(u for st in steps for u in st["uids"])
    short_uids = set(range(first_uid, first_uid + len(shorts)))
    long_uids = set(range(first_uid + len(shorts), first_uid + len(shorts) + len(longs)))
    times = collections.defaultdict(list)
    for st in steps:
        for uid in st["uids"]:
            times[uid].append(st["t"] - started)
    gaps = [b - a for uid in short_uids for a, b in zip(times[uid], times[uid][1:])]
    result = {
        **config, "elapsed_seconds": elapsed,
        "generated_tokens": sum(len(o["token_ids"]) for o in outputs),
        "steps": len(steps), "prefill_steps": sum(st["prefill"] for st in steps),
        "short_completion_s_max": max(times[u][-1] for u in short_uids),
        "short_completion_s_median": statistics.median(times[u][-1] for u in short_uids),
        "short_gap_ms_max": 1000 * max(gaps), "short_gap_ms_median": 1000 * statistics.median(gaps),
        "short_gap_ms_p99": 1000 * sorted(gaps)[int(0.99 * (len(gaps) - 1))],
        # first token is ready when the request's first decode step starts (after its last prefill chunk)
        "long_first_token_s": [min(st["t"] - started for st in steps if not st["prefill"] and u in st["uids"])
                               for u in sorted(long_uids)],
        "short_first_decode_s_median": statistics.median(
            min(st["t"] - started for st in steps if not st["prefill"] and u in st["uids"]) for u in short_uids),
        "step_log": [{"t": round(st["t"] - started, 6), "prefill": st["prefill"], "n": len(st["uids"]),
                      "extend_tokens": st["extend_tokens"]} for st in steps],
    }
    Path(out_path).write_text(json.dumps(result))


def interference_case() -> dict:
    import subprocess
    import sys

    import torch

    results = []
    for config in INTERFERENCE_CONFIGS:
        out_path = f"/tmp/{config['name']}.json"
        proc = subprocess.run([sys.executable, "/root/app.py", "interference_worker", json.dumps(config), out_path],
                              capture_output=True, text=True)
        results.append(json.loads(Path(out_path).read_text()) if proc.returncode == 0
                       else {**config, "error": f"exit {proc.returncode}", "stderr_tail": proc.stderr[-1500:]})
    return {"variant": "bf16", "configs": results, "gpu": torch.cuda.get_device_name(0),
            "measurement": "offline; per-request token times from scheduled-step timestamps (overlap scheduler)"}


@app.function(image=make_image("bf16"), **{**GPU_LIMITS, "timeout": 1800})
def interference_bf16() -> dict:
    return interference_case()


@app.function(image=make_image("bf16_testonly"), **CPU_LIMITS)
def unittest_testonly() -> dict:
    return unit_test_case("bf16_testonly")


@app.function(image=make_image("bf16_cachefix"), **CPU_LIMITS)
def unittest_cachefix() -> dict:
    return unit_test_case("bf16_cachefix")


NEEDLE_ABLATIONS = {  # experiment 7b: name -> (function, options)
    "bf16": ("bf16", {}),
    "fp8_sat": ("fp8_sat", {}),
    "bf16_naive": ("bf16", {"cache_type": "naive"}),
    "bf16_len2048": ("bf16", {"lengths": (2048,)}),
    "fp8_sat_capmatch": ("fp8_sat", {"num_page_override": BF16_W2_PAGES}),
    "bf16_uafdetect": ("bf16", {"detect_uaf": True}),  # experiment 9a
    "bf16_cachefix": ("bf16_cachefix", {}),  # experiment 9b
}


@app.function(image=make_image("bf16"), **GPU_LIMITS)
def eval_bf16() -> dict:
    return eval_case("bf16")


@app.function(image=make_image("fp8"), **GPU_LIMITS)
def eval_fp8() -> dict:
    return eval_case("fp8")


@app.function(image=make_image("fp8_sat"), **GPU_LIMITS)
def eval_fp8_sat() -> dict:
    return eval_case("fp8_sat")


@app.function(image=make_image("bf16"), **GPU_LIMITS)
def run_bf16(input_length: int, capture_trace: bool) -> dict:
    return run_case("bf16", input_length, capture_trace)


@app.function(image=make_image("fp8"), **GPU_LIMITS)
def run_fp8(input_length: int, capture_trace: bool) -> dict:
    return run_case("fp8", input_length, capture_trace)


@app.function(image=make_image("fp8_sat"), **GPU_LIMITS)
def run_fp8_sat(input_length: int, capture_trace: bool) -> dict:
    return run_case("fp8_sat", input_length, capture_trace)


@app.function(image=make_image("bf16"), **GPU_LIMITS)
def diagnose_bf16(input_length: int) -> dict:
    return diagnose_case("bf16", input_length)


@app.function(image=make_image("fp8"), **GPU_LIMITS)
def diagnose_fp8(input_length: int) -> dict:
    return diagnose_case("fp8", input_length)


@app.function(image=make_image("fp8_sat"), **GPU_LIMITS)
def diagnose_fp8_sat(input_length: int) -> dict:
    return diagnose_case("fp8_sat", input_length)


RUNNERS = {"bf16": run_bf16, "fp8": run_fp8, "fp8_sat": run_fp8_sat}
DIAGNOSERS = {"bf16": diagnose_bf16, "fp8": diagnose_fp8, "fp8_sat": diagnose_fp8_sat}


@app.local_entrypoint()
def main(variant: str = "bf16", input_length: int = 8000,
         repeat: int = 1, stage: str = "final", dry_run: bool = False):
    allowed = {"bench": (*VARIANTS, "fp8_sat_capmatch"), "interference": ("bf16",),
               "needle": tuple(NEEDLE_ABLATIONS), "unittest": TEST_IMAGES}.get(stage, VARIANTS)
    if variant not in allowed:
        raise ValueError(f"--variant must be one of {allowed}")
    if input_length not in INPUT_LENGTHS:
        raise ValueError(f"--input-length must be one of {INPUT_LENGTHS}")
    if repeat not in ((1, 2, 3, 4) if stage == "bench" else (1, 2)):
        raise ValueError("--repeat must be 1 or 2 (1-4 for bench)")
    if stage not in ("pilot", "final", "diagnose", "bench", "eval", "sweep", "needle", "unittest", "interference"):
        raise ValueError("--stage must be pilot, final, diagnose, bench, eval, sweep, needle, or unittest")
    if stage == "interference":
        directory = Path("results/long-context-probe/minisgl-prefill-decode-interference")
        directory.mkdir(parents=True, exist_ok=True)
        run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-interference-r{repeat}"
        result = interference_bf16.remote()
        result.update({"run_id": run_id, "repeat": repeat})
        (directory / f"{run_id}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {directory / run_id}.json")
        for c in result["configs"]:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k not in ("step_log", "stderr_tail")})
        if failed := [c["name"] for c in result["configs"] if "error" in c]:
            raise RuntimeError(f"interference configs failed: {failed}")
        return
    if stage == "unittest":
        if variant not in TEST_IMAGES:
            raise ValueError(f"unittest variants: {TEST_IMAGES}")
        directory = Path("results/long-context-probe/minisgl-prefix-cache-uaf")
        directory.mkdir(parents=True, exist_ok=True)
        run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-unittest-{variant}"
        result = {"bf16_testonly": unittest_testonly, "bf16_cachefix": unittest_cachefix}[variant].remote()
        result.update({"run_id": run_id, "patch_sha256": PATCH_SHA256.get(variant)})
        (directory / f"{run_id}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {directory / run_id}.json")
        print(result["stdout_tail"][-1200:])
        return
    if stage in ("sweep", "needle"):
        allowed = tuple(NEEDLE_ABLATIONS) if stage == "needle" else ("bf16", "fp8_sat")
        if variant not in allowed or dry_run:
            raise ValueError(f"{stage} variants: {allowed}; no dry run")
        folder = {"sweep": "fp8-kv-break-even", "needle": "fp8-kv-needle"}[stage]
        directory = Path("results/long-context-probe") / folder
        directory.mkdir(parents=True, exist_ok=True)
        run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{stage}-{variant}-r{repeat}"
        if stage == "sweep":
            result = {"bf16": sweep_bf16, "fp8_sat": sweep_fp8_sat}[variant].remote()
        else:
            image_variant, options = NEEDLE_ABLATIONS[variant]
            result = {"bf16": needle_bf16, "fp8_sat": needle_fp8_sat,
                      "bf16_cachefix": needle_bf16_cachefix}[image_variant].remote(**options)
            result["ablation"] = variant
            variant = image_variant
        result.update({"run_id": run_id, "repeat": repeat, "patch_sha256": PATCH_SHA256.get(variant)})
        (directory / f"{run_id}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {directory / run_id}.json")
        print({k: v for k, v in result.items() if k not in ("records", "cells")})
        return
    if stage == "eval":
        if dry_run:
            raise ValueError("eval has no dry run")
        directory = Path("results/long-context-probe/fp8-kv-accuracy-eval")
        directory.mkdir(parents=True, exist_ok=True)
        run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-eval-{variant}"
        result = {"bf16": eval_bf16, "fp8": eval_fp8, "fp8_sat": eval_fp8_sat}[variant].remote()
        result.update({"run_id": run_id, "patch_sha256": PATCH_SHA256.get(variant)})
        (directory / f"{run_id}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {directory / run_id}.json")
        print({k: v for k, v in result.items() if k != "records"})
        return
    if stage == "bench":
        if variant not in ("bf16", "fp8_sat", "fp8_sat_capmatch") or dry_run:
            raise ValueError("bench variants: bf16, fp8_sat, fp8_sat_capmatch; no dry run")
        directory = Path("results/long-context-probe/fp8-kv-capacity-w2")
        directory.mkdir(parents=True, exist_ok=True)
        run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-bench-{variant}-r{repeat}"
        if variant == "bf16":
            result = bench_bf16.remote()
        else:
            result = bench_fp8_sat.remote(BF16_W2_PAGES if variant == "fp8_sat_capmatch" else None)
            result["variant"] = variant
        variant = variant.removesuffix("_capmatch")
        result.update({"run_id": run_id, "repeat": repeat,
                       "patch_sha256": PATCH_SHA256.get(variant)})
        (directory / f"{run_id}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {directory / run_id}.json")
        print(json.dumps(result, indent=2))
        return
    if stage == "diagnose":
        if input_length != 1024 or dry_run:
            raise ValueError("diagnose uses 1024 input tokens and has no dry run")
        directory = Path("results/long-context-probe/fp8-kv-cache-feasibility/diagnose")
        directory.mkdir(parents=True, exist_ok=True)
        result = DIAGNOSERS[variant].remote(input_length)
        result["run_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        path = directory / f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{variant}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {path}")
        return
    if stage == "pilot" and (variant == "bf16" or input_length != 1024 or repeat != 1):
        raise ValueError("pilot is an FP8 variant, 1024 input tokens, repeat 1")
    if stage == "final" and input_length != 8000:
        raise ValueError("final uses 8000 input tokens")
    capture_trace = stage == "final" and repeat == 1
    if dry_run:
        print(json.dumps({"stage": stage, "variant": variant, "input_length": input_length,
                          "output_tokens": OUTPUT_TOKENS, "repeat": repeat,
                          "capture_trace": capture_trace, "engine_commit": BASE_SHA,
                          "patch_sha256": PATCH_SHA256.get(variant),
                          "pins": PINS, "estimated_max_usd": 0.34,
                          "gpu_launched": False}, indent=2))
        return

    run_id = (f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-"
              f"{uuid.uuid4().hex[:8]}-{stage}-{variant}-{input_length}-r{repeat}")
    directory = Path("results/long-context-probe/fp8-kv-cache-feasibility") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        result = RUNNERS[variant].remote(input_length, capture_trace)
        record = result["record"]
        record["status"] = "completed" if record["valid_count"] else "qualified"
        if result["trace_gz"]:
            (directory / "trace.json").write_bytes(gzip.decompress(result["trace_gz"]))
            record["trace_file"] = "trace.json"
    except Exception as exc:
        record = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    record.update({"run_id": run_id, "repeat": repeat, "stage": stage, "variant": variant,
                   "input_length_requested": input_length, "capture_trace": capture_trace,
                   "started_utc": started,
                   "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    (directory / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {directory / 'run.json'}")
    summary = {k: v for k, v in record.items() if k != "quality_fixtures"}
    print(json.dumps(summary, indent=2))
    if record["status"] == "failed":
        raise RuntimeError(record["error"])


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "interference_worker":
    interference_worker(json.loads(sys.argv[2]), sys.argv[3])
