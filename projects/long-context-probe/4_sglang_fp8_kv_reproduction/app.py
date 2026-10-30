"""Does full SGLang's FP8 KV path reproduce the mini-SGLang e4m3 overflow on Qwen3-0.6B?

Runs the official SGLang image unmodified (no engine source copied here). One
container runs each KV dtype in its own subprocess, so every engine starts on a
clean GPU. The `sat` image applies a one-line saturating cast to the installed
MHATokenToKVPool.set_kv_buffer (mirroring open SGLang PR #41732's MLA change).
"""

import base64
import datetime as dt
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import modal

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.9-cu129-amd64"
MODEL_DIR = "/root/huggingface/Qwen3-0.6B"
MODEL_REVISION_EXPECTED = "c1899de289a04d12100db370d81485cdf75e47ca"
SEED = 80001000
RANDOM_LENGTHS = (1024, 8000)
OUTPUT_TOKENS = 1000
QUALITY_OUTPUT_TOKENS = 128
QUALITY_PROMPTS = (  # same fixtures as experiment 3
    "The capital of France is",
    "Explain why the sky is blue in two sentences.",
    "def fibonacci(n):\n",
    "Write a haiku about autumn leaves.",
    "List three differences between TCP and UDP.",
    "Translate to French: The weather is nice today.",
    "What is 17 multiplied by 23? Show your work.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
)
LOG_PATTERNS = ("KV Cache is allocated", "scaling factors", "attention backend", "NaN", "KVPROBE")
RESULTS = Path("results/long-context-probe/sglang-fp8-kv-reproduction")

# Replaces the plain cast in MHATokenToKVPool.set_kv_buffer (0.5.9) with clamp-then-cast.
SAT_OLD = "            cache_k = cache_k.to(self.dtype)\n            cache_v = cache_v.to(self.dtype)\n"
SAT_NEW = ("            if self.dtype == torch.float8_e4m3fn:  # study patch: saturate to +-448\n"
           "                fmax = torch.finfo(self.dtype).max\n"
           "                cache_k, cache_v = cache_k.clamp(-fmax, fmax), cache_v.clamp(-fmax, fmax)\n"
           + SAT_OLD)
SAT_SCRIPT = f"""
import sglang.srt.mem_cache.memory_pool as m, pathlib
p = pathlib.Path(m.__file__); s = p.read_text()
old, new = {SAT_OLD!r}, {SAT_NEW!r}
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, new)); print("patched", p)
"""

# K-range probe: log layer-0 K/V magnitude right before the FP8 cast (eager calls only).
PROBE_NEW = ("            if layer_id == 0 and not torch.cuda.is_current_stream_capturing():  # study probe\n"
             "                _a = cache_k.float().abs()\n"
             "                print(f'KVPROBE layer0 tokens={cache_k.shape[0]} k_max={_a.max().item():.1f} '\n"
             "                      f'k_over448={int((_a > 448).sum())} k_over464={int((_a > 464).sum())} '\n"
             "                      f'v_max={cache_v.float().abs().max().item():.2f}', flush=True)\n"
             + SAT_OLD)
PROBE_SCRIPT = f"""
import sglang.srt.mem_cache.memory_pool as m, pathlib
p = pathlib.Path(m.__file__); s = p.read_text()
old, new = {SAT_OLD!r}, {PROBE_NEW!r}
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, new)); print("probed", p)
"""

base_image = (modal.Image.from_registry(SGLANG_IMAGE).entrypoint([])
              .env({"HF_HOME": "/root/.cache/huggingface"}))
# Dockerfile RUN lines cannot contain newlines, so ship the script base64-encoded.
sat_image = base_image.run_commands(
    f"python -c \"import base64; exec(base64.b64decode('{base64.b64encode(SAT_SCRIPT.encode()).decode()}'))\"")

probe_image = base_image.run_commands(
    f"python -c \"import base64; exec(base64.b64decode('{base64.b64encode(PROBE_SCRIPT.encode()).decode()}'))\"")

app = modal.App("sglang-fp8-kv-reproduction")
weights = modal.Volume.from_name("llm-bench-hf-cache")
GPU_LIMITS = dict(gpu="L4", cpu=4, memory=16384, max_containers=1,
                  startup_timeout=300, timeout=1800, scaledown_window=60,
                  volumes={"/root/huggingface": weights})


def worker(variant: str, out_path: str) -> None:
    """Runs inside a subprocess: one engine, all prompts, results to JSON.

    A `_cudacore` suffix forces FlashInfer's CUDA-core decode kernel; SGLang otherwise
    always uses tensor-core decode for FP8 KV (flashinfer_backend.should_use_tensor_core).
    """
    import os

    kv_cache_dtype = variant.removesuffix("_cudacore")
    if variant.endswith("_cudacore"):
        os.environ["SGLANG_FLASHINFER_USE_TENSOR_CORE"] = "false"
    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    engine = sgl.Engine(model_path=MODEL_DIR, kv_cache_dtype=kv_cache_dtype,
                        disable_radix_cache=True, context_length=9216,
                        mem_fraction_static=0.8, cuda_graph_max_bs=8, log_level="info")
    rng = random.Random(SEED)
    full = [rng.randint(0, 10000) for _ in range(max(RANDOM_LENGTHS))]
    record = {"variant": variant, "kv_cache_dtype": kv_cache_dtype, "sglang": sgl.__version__,
              "decode_use_tensor_cores_env": os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE"),
              "attention_backend": getattr(engine.server_args, "attention_backend", None),
              "random": [], "quality": []}
    for length in RANDOM_LENGTHS:
        prompt = full[:length]  # same seeded prefix scheme as experiments 2-3
        started = time.perf_counter()
        out = engine.generate(input_ids=prompt, sampling_params={
            "temperature": 0.0, "max_new_tokens": OUTPUT_TOKENS, "ignore_eos": True})
        elapsed = time.perf_counter() - started
        ids = out["output_ids"]
        record["random"].append({
            "input_tokens": length,
            "prompt_sha256": hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(),
            "output_tokens": len(ids), "first_32_ids": ids[:32], "token0_count": ids.count(0),
            "output_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "elapsed_seconds_unpaired": elapsed})
    outs = engine.generate(input_ids=[tokenizer.encode(p) for p in QUALITY_PROMPTS], sampling_params={
        "temperature": 0.0, "max_new_tokens": QUALITY_OUTPUT_TOKENS, "ignore_eos": True})
    for prompt, out in zip(QUALITY_PROMPTS, outs):
        ids = out["output_ids"]
        record["quality"].append({"prompt": prompt, "token_ids": ids, "token0_count": ids.count(0),
                                  "text": tokenizer.decode(ids)})
    engine.shutdown()
    Path(out_path).write_text(json.dumps(record))


def run_variants(variants: list) -> dict:
    import torch

    results = {}
    for kv in variants:
        out_path, log_path = f"/tmp/{kv}.json", f"/tmp/{kv}.log"
        with open(log_path, "w") as log:
            code = subprocess.run([sys.executable, "/root/app.py", "worker", kv, out_path],
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        log_lines = [l.strip()[:300] for l in Path(log_path).read_text().splitlines()
                     if any(p in l for p in LOG_PATTERNS)]
        tail = Path(log_path).read_text().splitlines()[-25:]
        results[kv] = (json.loads(Path(out_path).read_text()) if code == 0 and Path(out_path).exists()
                       else {"kv_cache_dtype": kv, "error": f"worker exit {code}", "log_tail": tail})
        results[kv]["log_lines"] = log_lines
    return {"variants": results, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "model_revision_expected_not_verified": MODEL_REVISION_EXPECTED}


@app.function(image=base_image, **GPU_LIMITS)
def run_base(variants: list) -> dict:
    return run_variants(variants)


@app.function(image=sat_image, **GPU_LIMITS)
def run_sat(variants: list) -> dict:
    return run_variants(variants)


@app.function(image=probe_image, **GPU_LIMITS)
def run_probe(variants: list) -> dict:
    return run_variants(variants)


@app.function(image=base_image, **{**GPU_LIMITS, "timeout": 900})
def run_cudacore(variants: list) -> dict:
    return run_variants(variants)


@app.local_entrypoint()
def main(image: str = "base", variants: str = "auto,fp8_e4m3,fp8_e5m2", dry_run: bool = False):
    chosen = variants.split(",")
    allowed = {"auto", "fp8_e4m3", "fp8_e5m2", "fp8_e4m3_cudacore"}
    if image not in ("base", "sat", "probe", "cudacore") or not set(chosen) <= allowed:
        raise ValueError(f"--image base|sat|probe|cudacore; --variants from {sorted(allowed)}")
    estimate = round((300 + 1800 + 60) / 60 * 0.016 * 1.4, 2)  # 4 CPU / 16 GiB instead of 2 / 8
    if dry_run:
        print(json.dumps({"image": image, "sglang_image": SGLANG_IMAGE, "variants": chosen,
                          "estimated_max_usd": estimate, "gpu_launched": False}, indent=2))
        return
    run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{image}-{'-'.join(chosen)}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    result = {"base": run_base, "sat": run_sat, "probe": run_probe,
              "cudacore": run_cudacore}[image].remote(chosen)
    result.update({"run_id": run_id, "image": image, "sglang_image": SGLANG_IMAGE,
                   "sat_patch": SAT_NEW if image == "sat" else None})
    path = RESULTS / f"{run_id}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {path}")
    for kv, r in result["variants"].items():
        if "error" in r:
            print(kv, r["error"], *r["log_tail"][-8:], sep="\n  ")
            continue
        print(kv, r["attention_backend"], [(x["input_tokens"], x["token0_count"], x["output_sha256"][:12],
                                            round(x["elapsed_seconds_unpaired"], 3)) for x in r["random"]],
              "quality token0:", [q["token0_count"] for q in r["quality"]])
        print("  logs:", *r["log_lines"][:12], sep="\n    ")


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "worker":
    worker(sys.argv[2], sys.argv[3])
