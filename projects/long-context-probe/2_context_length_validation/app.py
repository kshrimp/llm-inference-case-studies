"""Offline input-length validation on pinned, unmodified mini-SGLang.

No upstream engine source is copied into this repository. The baseline is
installed from one exact Git commit (mini-SGLang is MIT licensed).
"""

import datetime as dt
import hashlib
import json
import random
import re
import subprocess
import time
import uuid
from pathlib import Path

import modal

BASE_REPO = "https://github.com/sgl-project/mini-sglang.git"
BASE_SHA = "9a91cfafe754aa85daee49998176275667eb58f2"
MODEL_DIR = "/root/huggingface/Qwen3-0.6B"
MODEL_REVISION_EXPECTED = "c1899de289a04d12100db370d81485cdf75e47ca"
INPUT_LENGTHS = (1024, 4096, 8000)
OUTPUT_TOKENS = 1000
TRACE_OUTPUT_TOKENS = 128
TRACE_VOLUME = "long-context-length-traces"
SEED = 80001000


def validate_source(repo: str, sha: str) -> None:
    if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", repo):
        raise ValueError("source repository must be an HTTPS GitHub URL")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise ValueError("source commit must be a full 40-character SHA")


def make_image(repo: str, sha: str):
    validate_source(repo, sha)
    return (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
        .apt_install("git", "build-essential")
        .env({"CUDA_HOME": "/usr/local/cuda", "TVM_FFI_CACHE_DIR": "/cache/tvm-ffi",
              "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer"})
        .run_commands(
            f"git clone {repo} /opt/mini-sglang"
            f" && cd /opt/mini-sglang && git checkout {sha}"
            " && python -m pip install -e . && python -m pip install torch-c-dlpack-ext"
        )
    )


app = modal.App("long-context-length-validation")
weights = modal.Volume.from_name("llm-bench-hf-cache")
jit_cache = modal.Volume.from_name("tiny-engine-jit-cache")
traces = modal.Volume.from_name(TRACE_VOLUME, create_if_missing=True)

GPU_LIMITS = dict(
    gpu="L4", cpu=2, memory=8192, max_containers=1,
    startup_timeout=300, timeout=900, scaledown_window=60,
    volumes={"/root/huggingface": weights, "/cache": jit_cache, "/traces": traces},
)

def run_case(input_length: int, capture_trace: bool, run_id: str) -> dict:
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    if input_length not in INPUT_LENGTHS:
        raise ValueError(f"input_length must be one of {INPUT_LENGTHS}")
    if not Path(MODEL_DIR, "config.json").is_file():
        raise FileNotFoundError("Model snapshot missing from llm-bench-hf-cache")

    rng = random.Random(SEED)
    prompt = [rng.randint(0, 10000) for _ in range(input_length)]
    prompt_hash = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
    llm = LLM(MODEL_DIR, max_seq_len_override=9216, max_extend_tokens=16384,
              cuda_graph_max_bs=1, page_size=256)
    llm.generate(["Warmup"], SamplingParams(temperature=0.0, max_tokens=1))
    target_tokens = TRACE_OUTPUT_TOKENS if capture_trace else OUTPUT_TOKENS
    params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=target_tokens)
    torch.cuda.synchronize()
    started = time.perf_counter()
    trace_file = None
    if capture_trace:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                 torch.profiler.ProfilerActivity.CUDA]) as profiler:
            outputs = llm.generate([prompt], params)
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        trace_file = f"{run_id}.json"
        profiler.export_chrome_trace(f"/traces/{trace_file}")
        traces.commit()
    else:
        outputs = llm.generate([prompt], params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    token_ids = outputs[0]["token_ids"]
    return {
        "engine_repo": BASE_REPO, "engine_commit": BASE_SHA,
        "input_tokens": len(prompt), "prompt_sha256": prompt_hash,
        "measurement": "trace_128" if capture_trace else "unprofiled_1000",
        "requested_output_tokens": target_tokens, "actual_output_tokens": len(token_ids),
        "output_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
        "elapsed_seconds": elapsed, "valid_count": len(token_ids) == target_tokens,
        "trace_volume": TRACE_VOLUME if trace_file else None,
        "trace_file": trace_file,
        "sampling": {"temperature": 0.0, "ignore_eos": True},
        "model_revision_expected_not_verified": MODEL_REVISION_EXPECTED,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }


@app.function(image=make_image(BASE_REPO, BASE_SHA), **GPU_LIMITS)
def run_base(input_length: int, capture_trace: bool, run_id: str) -> dict:
    return run_case(input_length, capture_trace, run_id)


@app.local_entrypoint()
def main(input_length: int = 8000, repeat: int = 1, capture_trace: bool = False,
         dry_run: bool = False):
    if input_length not in INPUT_LENGTHS:
        raise ValueError(f"--input-length must be one of {INPUT_LENGTHS}")
    if repeat not in (1, 2):
        raise ValueError("--repeat must be 1 or 2")
    if capture_trace and repeat != 1:
        raise ValueError("trace runs use --repeat 1; unprofiled repeats stay separate")
    if dry_run:
        print(json.dumps({"experiment": "context_length_validation", "input_length": input_length,
                          "output_tokens": TRACE_OUTPUT_TOKENS if capture_trace else OUTPUT_TOKENS,
                          "capture_trace": capture_trace, "repeat": repeat,
                          "engine_commit": BASE_SHA,
                          "estimated_max_usd": 0.34, "gpu_launched": False}, indent=2))
        return

    run_id = (f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-"
              f"{uuid.uuid4().hex[:8]}-length-{input_length}-"
              f"{'trace' if capture_trace else 'measure'}-r{repeat}")
    directory = Path("results/long-context-probe/context-length-validation") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        result = run_base.remote(input_length, capture_trace, run_id)
        record = {"status": "completed" if result["valid_count"] else "qualified", **result}
    except Exception as exc:
        record = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    record.update({"run_id": run_id, "repeat": repeat, "capture_trace_requested": capture_trace,
                   "input_length_requested": input_length, "started_utc": started,
                   "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    (directory / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {directory / 'run.json'}")
    if record.get("trace_file"):
        print(f"Download trace: uv run modal volume get {TRACE_VOLUME} {record['trace_file']} {directory / 'trace.json'}")
    print(json.dumps(record, indent=2))
    if record["status"] == "failed":
        raise RuntimeError(record["error"])
