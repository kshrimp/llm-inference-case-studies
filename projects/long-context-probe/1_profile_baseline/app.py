"""Bounded offline long-context probe for pinned mini-SGLang (MIT).

Engine code is installed from upstream; none is copied into this project.
"""

import datetime as dt
import json
import random
import subprocess
import time
import uuid
from pathlib import Path

import modal

MINI_SHA = "9a91cfafe754aa85daee49998176275667eb58f2"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_DIR = "/root/huggingface/Qwen3-0.6B"
TRACE_VOLUME = "long-context-probe-traces"
OUTPUT_TOKENS = {"smoke": 8, "profile": 32, "full": 1000, "full-profile": 1000}

app = modal.App("long-context-probe")
weights = modal.Volume.from_name("llm-bench-hf-cache")
jit_cache = modal.Volume.from_name("tiny-engine-jit-cache")
traces = modal.Volume.from_name(TRACE_VOLUME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda", "TVM_FFI_CACHE_DIR": "/cache/tvm-ffi",
          "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer"})
    .run_commands(
        "git clone https://github.com/sgl-project/mini-sglang.git /opt/mini-sglang"
        f" && cd /opt/mini-sglang && git checkout {MINI_SHA}"
        " && python -m pip install -e . && python -m pip install torch-c-dlpack-ext"
    )
)


@app.function(
    image=image, gpu="L4", cpu=2, memory=8192, max_containers=1,
    startup_timeout=300, timeout=900, scaledown_window=60,
    volumes={"/root/huggingface": weights, "/cache": jit_cache, "/traces": traces},
)
def run_probe(mode: str, run_id: str) -> dict:
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    if mode not in OUTPUT_TOKENS:
        raise ValueError("mode must be smoke, profile, full, or full-profile")
    if not Path(MODEL_DIR, "config.json").is_file():
        raise FileNotFoundError("Pinned model snapshot missing from llm-bench-hf-cache")

    rng = random.Random(80001000)
    prompt = [rng.randint(0, 10000) for _ in range(8000)]
    llm = LLM(MODEL_DIR, max_seq_len_override=9216, max_extend_tokens=16384,
              cuda_graph_max_bs=1, page_size=256)
    llm.generate(["Warmup"], SamplingParams(temperature=0.1, max_tokens=1))
    params = SamplingParams(temperature=0.6, ignore_eos=True,
                            max_tokens=OUTPUT_TOKENS[mode])
    trace_path = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    if mode in ("profile", "full-profile"):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                 torch.profiler.ProfilerActivity.CUDA]) as prof:
            outputs = llm.generate([prompt], params)
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        trace_path = f"/traces/{run_id}.json"
        prof.export_chrome_trace(trace_path)
        traces.commit()
    else:
        outputs = llm.generate([prompt], params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    actual = len(outputs[0]["token_ids"])
    return {
        "mode": mode, "run_id": run_id, "input_tokens": len(prompt),
        "requested_output_tokens": params.max_tokens, "actual_output_tokens": actual,
        "elapsed_seconds": elapsed, "valid_count": actual == params.max_tokens,
        "trace_volume": TRACE_VOLUME if trace_path else None,
        "trace_file": Path(trace_path).name if trace_path else None,
        "engine_commit": MINI_SHA, "model_revision": MODEL_REVISION,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }


@app.local_entrypoint()
def main(mode: str = "smoke"):
    if mode not in OUTPUT_TOKENS:
        raise ValueError("--mode must be smoke, profile, full, or full-profile")
    run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}-{mode}"
    directory = Path("results/long-context-probe") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        result = run_probe.remote(mode, run_id)
        record = {"status": "completed" if result["valid_count"] else "qualified", **result}
    except Exception as exc:
        record = {"status": "failed", "mode": mode, "run_id": run_id,
                  "error": f"{type(exc).__name__}: {exc}"}
    record["started_utc"] = started
    record["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    record["source_revision"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip() or None
    (directory / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {directory / 'run.json'}")
    if record.get("trace_file"):
        print(f"Download trace: uv run modal volume get {TRACE_VOLUME} {record['trace_file']} {directory / 'trace.json'}")
    if record["status"] == "failed":
        raise RuntimeError(record["error"])
    print(json.dumps(record, indent=2))
