"""Bounded Modal runner for the Qwen3-Omni on Modal project (README.md).

Stages: `cpu-check` (CPU) builds the image at the pinned SGLang-Omni
commit, records the installed stack, and resolves the configs the experiments
need, including whether `--thinker.engine.kv_cache_dtype fp8_e4m3` is accepted
(the CPU half of experiment 2's condition C gate). `download-fp8` fills the
weights Volume with the community FP8 checkpoint at its pinned revision and
verifies every file against Hugging Face metadata. `anatomy` is one experiment 1
container (1_long_context_anatomy/README.md): BF16 text-only server on one H100.
`fp8-smoke` is experiment 2a (2_fp8_thinker/README.md): B, C, then A on one H100.
`fp8-backends` is the redesigned 2b: FP8 backend sessions plus a BF16 drift check.
`fp8-combined` is 2b-2: DeepGEMM dense + Triton MoE, a needle-retrieval quality set, and 8 concurrent requests.
`capacity-scan` is experiment 4a: memory fraction and closed-loop concurrency scan, BF16 and FP8.
`fp8-tune` is 2c: SGLang's own block-FP8 tuning function (v0.5.22 script, unmodified; search space
filtered to BLOCK_SIZE_M = 16 for decode sizes) for the thinker's two dense shapes, then one FP8 session with Triton dense + Triton MoE using the tuned configs.
`capacity-sweep` is experiment 4b: open-loop Poisson rates against the frozen pass criteria.

No upstream source is copied here. sglang-omni (Apache-2.0) is installed from one
exact commit, following that commit's docker/Dockerfile dependency recipe. The
image recipe mirrors projects/omni-pipeline-probe/app.py, with the new pin.

Usage: uv run modal run projects/qwen3-omni-on-modal/app.py --stage cpu-check
"""

import datetime as dt
import json
import re
import subprocess
import uuid
from pathlib import Path

import modal

OMNI_REPO = "https://github.com/sgl-project/sglang-omni.git"
OMNI_SHA = "305273a673c281f2d214e1bed457a4b5b8895740"  # main, 2026-10-27
# Same digest as the pinned commit's docker/Dockerfile SGLANG_IMAGE argument.
BASE_IMAGE = ("lmsysorg/sglang@sha256:"
              "5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e")
SGLANG_PY = "/opt/sglang/bin/python3"
WEIGHTS_MOUNT = "/weights"
BF16_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
BF16_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
BF16_DIR = f"{WEIGHTS_MOUNT}/Qwen3-Omni-30B-A3B-Instruct"  # filled by the omni probe
FP8_ID = "marksverdhei/Qwen3-Omni-30B-A3B-FP8"  # community conversion, not an official Qwen release
FP8_REVISION = "042db39a102171d5994b72ab717fade92e80061a"
FP8_DIR = f"{WEIGHTS_MOUNT}/Qwen3-Omni-30B-A3B-FP8"
FP8_CONFIG = "/opt/sglang-omni/examples/configs/qwen3_omni_fp8_colocated.yaml"
RESULTS = Path("results/qwen3-omni-on-modal/0_setup")
ANATOMY_RESULTS = Path("results/qwen3-omni-on-modal/1_anatomy")
FP8_SMOKE_RESULTS = Path("results/qwen3-omni-on-modal/2a_fp8_smoke")
FP8_BACKEND_RESULTS = Path("results/qwen3-omni-on-modal/2b_fp8_backends")
FP8_COMBINED_RESULTS = Path("results/qwen3-omni-on-modal/2b2_fp8_combined")
CAPACITY_RESULTS = Path("results/qwen3-omni-on-modal/4a_capacity_scan")
TUNE_RESULTS = Path("results/qwen3-omni-on-modal/2c_fp8_tune")
SWEEP_RESULTS = Path("results/qwen3-omni-on-modal/4b_capacity_sweep")
LOCAL_TEXT = Path("results/qwen3-omni-on-modal/inputs/pg132_raw.txt")
STAGES = ("cpu-check", "download-fp8", "anatomy", "fp8-smoke", "fp8-backends", "fp8-combined", "capacity-scan",
          "fp8-tune", "capacity-sweep")

for pin in (OMNI_SHA, BF16_REVISION, FP8_REVISION):
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise ValueError(f"pin must be a full 40-character SHA: {pin}")

# Mirrors the pinned Dockerfile: project dependencies with its override list and
# the DAC packages, then flashinfer-python 0.7.1 without dependencies. Skipped:
# UCX build (RDMA transport; one-GPU colocated runs do not use it), the
# audar-tts / fun-cosyvoice3 / minicpm-o extras and qwen-tts (other models), and
# the upstream entrypoint that pulls `main` at container start.
image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .apt_install("sox", "git")
    .run_commands(
        f"git clone --filter=blob:none {OMNI_REPO} /opt/sglang-omni"
        f" && git -C /opt/sglang-omni checkout {OMNI_SHA}",
        f"{SGLANG_PY} -c 'import tomllib; print(\"\\n\".join(tomllib.load(open("
        "\"/opt/sglang-omni/pyproject.toml\", \"rb\"))[\"tool\"][\"uv\"][\"override-dependencies\"]))'"
        " > /tmp/omni-overrides.txt",
        f"uv pip install --python {SGLANG_PY} --no-build-isolation"
        " --overrides /tmp/omni-overrides.txt -r /opt/sglang-omni/pyproject.toml"
        " descript-audiotools==0.7.2 descript-audio-codec==1.0.0",
        f"uv pip install --python {SGLANG_PY} --no-deps --reinstall flashinfer-python==0.7.1",
        f"uv pip install --python {SGLANG_PY} --no-deps --no-build-isolation -e /opt/sglang-omni",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "FLASHINFER_WORKSPACE_BASE": "/root",
          "HF_HUB_DISABLE_TELEMETRY": "1"})
)

app = modal.App("qwen3-omni-on-modal")
weights = modal.Volume.from_name("omni-hf-cache", create_if_missing=True)
jit_cache = modal.Volume.from_name("qwen3-omni-jit-cache", create_if_missing=True)
# Raw GPU-run records are also written here, so a dropped local client loses nothing.
results_vol = modal.Volume.from_name("qwen3-omni-results", create_if_missing=True)

# Ceilings match budget.py row 0: 5 min startup + 60 min run + 1 min idle.
CPU_LIMITS = dict(cpu=4, memory=8192, max_containers=1, retries=0, timeout=3600,
                  scaledown_window=60)


def sh(cmd: list[str], timeout: int = 600, keep: int = 20000) -> dict:
    import os
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, "COLUMNS": "200"})
    return {"cmd": cmd, "returncode": proc.returncode,
            "stdout": proc.stdout[-keep:], "stderr": proc.stderr[-keep:]}


def resolve(*args: str) -> dict:
    """`sgl-omni config resolve` with the given serve-style arguments, as a diff."""
    return sh(["sgl-omni", "config", "resolve", *args, "--show", "diff"])


@app.function(image=image, volumes={WEIGHTS_MOUNT: weights}, **CPU_LIMITS)
def cpu_check() -> dict:
    """Installed stack, pin check, and the resolved configs for experiments 1 and 2."""
    versions = sh([SGLANG_PY, "-c", (
        "import importlib.metadata as m, json, sys\n"
        "out = {'python': sys.version.split()[0]}\n"
        "for p in ('torch', 'sglang', 'sglang-omni', 'transformers', 'flashinfer-python',"
        " 'flash-attn-4', 'sgl-kernel', 'deep-gemm', 'triton'):\n"
        "    try: out[p] = m.version(p)\n"
        "    except m.PackageNotFoundError: out[p] = None\n"
        "print(json.dumps(out))")])
    # torch >= 2.13 saturates eager BF16 -> e4m3 casts (long-context experiment 8).
    cast = sh([SGLANG_PY, "-c", (
        "import torch, json\n"
        "x = torch.tensor([516.0, -516.0, 448.0], dtype=torch.bfloat16)\n"
        "y = x.to(torch.float8_e4m3fn).float().tolist()\n"
        "print(json.dumps({'in': [516.0, -516.0, 448.0], 'out_cpu': y}))")])
    head = sh(["git", "-C", "/opt/sglang-omni", "rev-parse", "HEAD"])
    bf16_ref = BF16_DIR if Path(BF16_DIR, "config.json").is_file() else f"{BF16_ID}"
    fp8_ref = FP8_DIR if Path(FP8_DIR, "config.json").is_file() else f"{FP8_ID}"
    text_common = ["--text-only", "--thinker.engine.disable_radix_cache", "true"]
    return {
        "versions": versions, "cast_probe": cast,
        "omni_head": head["stdout"].strip(),
        "omni_head_matches_pin": head["stdout"].strip() == OMNI_SHA,
        "bf16_weights_on_volume": Path(BF16_DIR, "config.json").is_file(),
        "fp8_weights_on_volume": Path(FP8_DIR, "config.json").is_file(),
        "resolve": {
            # Experiment 1 and condition A server.
            "bf16_text": resolve("--model-path", bf16_ref, *text_common),
            # Condition B: upstream FP8 config, text-only.
            "fp8_text": resolve("--config", FP8_CONFIG, "--model-path", fp8_ref, *text_common),
            # Condition C, CPU gate: is the KV dtype override accepted for the thinker?
            "fp8_text_fp8kv": resolve("--config", FP8_CONFIG, "--model-path", fp8_ref,
                                      *text_common, "--thinker.engine.kv_cache_dtype",
                                      "fp8_e4m3"),
        },
        "serve_help": sh(["sgl-omni", "serve", "--help"], keep=200_000),
    }


@app.function(image=image, volumes={WEIGHTS_MOUNT: weights}, cpu=4, memory=8192,
              max_containers=1, retries=0, timeout=3600, scaledown_window=60)
def download_fp8() -> dict:
    """Download the pinned FP8 revision and verify sizes and LFS SHA-256 against the Hub."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(FP8_ID, revision=FP8_REVISION, files_metadata=True)
    if info.sha != FP8_REVISION:
        raise RuntimeError(f"Hub resolved {info.sha}, expected {FP8_REVISION}")
    expected = {s.rfilename: {"size": s.size, "sha256": s.lfs.sha256 if s.lfs else None}
                for s in info.siblings}
    started = dt.datetime.now(dt.timezone.utc)
    snapshot_download(FP8_ID, revision=FP8_REVISION, local_dir=FP8_DIR, max_workers=8)
    weights.commit()

    def check(name: str) -> dict:
        path = Path(FP8_DIR, name)
        row = {"file": name, "expected_size": expected[name]["size"],
               "size": path.stat().st_size if path.is_file() else None,
               "expected_sha256": expected[name]["sha256"], "sha256": None}
        if row["size"] is not None and row["expected_sha256"]:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(16 << 20), b""):
                    digest.update(block)
            row["sha256"] = digest.hexdigest()
        row["ok"] = (row["size"] == row["expected_size"]
                     and (row["expected_sha256"] is None or row["sha256"] == row["expected_sha256"]))
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(check, sorted(expected)))
    return {
        "model_id": FP8_ID, "revision": FP8_REVISION, "local_dir": FP8_DIR,
        "label": "community FP8 conversion, not an official Qwen release",
        "started_utc": started.isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "file_count": len(files), "total_bytes": sum(f["size"] or 0 for f in files),
        "all_ok": all(f["ok"] for f in files),
        "lfs_hash_checked": sum(f["sha256"] is not None for f in files),
        "files": files,
    }


# ---- Experiment 1: long-context anatomy (BF16, text-only, batch 1) ----------

PORT = 8000
SERVED_MODEL = "qwen3-omni"
TEXT_SHA256 = "2301ae1ce20875381a316b2fd0ac5ada00f71bacdc732458a11b3674b358e360"  # same pin as the omni probe
TEXT_START_MARKER = "5, 6. _The Moral Law_ causes"
# The chat API has no ignore_eos, so both prompts are copy tasks whose text is
# longer than the output cap: a valid request stops on length at exactly 1,000.
INSTRUCTION = "Repeat this text exactly:\n\n"
MAX_TOKENS = 1000
PROMPTS = {"L": 8000, "S": 1150}  # target server-reported prompt tokens; S is the control
PROMPT_TOLERANCE = 10
REQUEST_DEADLINE_S = 300
SERVER_READY_DEADLINE_S = 900
# Ceilings match budget.py row 1: 10 min startup + 30 min run + 2 min idle.
GPU_LIMITS = dict(gpu="H100!", cpu=8, memory=65536, max_containers=1, retries=0,
                  startup_timeout=600, timeout=1800, scaledown_window=120)
ANATOMY_CMD = [
    "sgl-omni", "serve", "--model-path", BF16_DIR, "--text-only",
    "--host", "127.0.0.1", "--port", str(PORT), "--model-name", SERVED_MODEL,
    "--thinker.engine.disable_radix_cache", "true",
]
ANATOMY_ENV = {"PYTHONUNBUFFERED": "1", "SGLANG_OMNI_WARMUP_TIMEOUT": "600",
               "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer", "TRITON_CACHE_DIR": "/cache/triton",
               "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor"}
# 2 warmups, then L and S interleaved 5 times each (untraced).
MEASURE_ORDER = ["L", "S"] * 5

anatomy_image = (
    image.run_commands(f"uv pip install --python {SGLANG_PY} py-spy==0.4.1")
    .add_local_file(str(LOCAL_TEXT), "/inputs/pg132_raw.txt")
)


def stream_chat(rid: str, text: str, max_tokens: int, deadline_s: float | None = None) -> dict:
    """One streaming request. Times are client monotonic ns relative to dispatch.

    `deadline_s` bounds the whole request (httpx's timeout bounds each read only)."""
    import time

    import httpx

    body = {"model": SERVED_MODEL, "messages": [{"role": "user", "content": text}],
            "modalities": ["text"], "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "request_id": rid}
    rec = {"request_id": rid, "status": "failed", "http_status": None, "max_tokens": max_tokens,
           "dispatch_wall_ns": time.time_ns(), "ttft_s": None, "completion_s": None,
           "chunk_offsets_us": [], "finish_reason": None, "usage": None, "text": "",
           "error": None}
    t0 = time.monotonic_ns()
    try:
        with httpx.Client(timeout=REQUEST_DEADLINE_S, trust_env=False) as client:
            with client.stream("POST", f"http://127.0.0.1:{PORT}/v1/chat/completions",
                               json=body) as resp:
                rec["http_status"] = resp.status_code
                if resp.status_code != 200:
                    rec["error"] = resp.read().decode(errors="replace")[:2000]
                    return rec
                for line in resp.iter_lines():
                    if deadline_s is not None and time.monotonic_ns() - t0 > deadline_s * 1e9:
                        rec["error"] = f"deadline {deadline_s} s"
                        return rec
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        rec["usage"] = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            now = time.monotonic_ns()
                            if rec["ttft_s"] is None:
                                rec["ttft_s"] = (now - t0) / 1e9
                            rec["chunk_offsets_us"].append((now - t0) // 1000)
                            rec["text"] += piece
                        if choice.get("finish_reason"):
                            rec["finish_reason"] = choice["finish_reason"]
        rec["status"] = "completed" if rec["text"] else "empty"
    except Exception as exc:  # keep the record on timeout, reset, or bad JSON
        rec["error"] = repr(exc)[:2000]
    finally:
        rec["completion_s"] = (time.monotonic_ns() - t0) / 1e9
    return rec


def valid(rec: dict) -> list[str]:
    """Reasons a measured request is invalid; empty means valid."""
    reasons = []
    if rec["status"] != "completed":
        reasons.append(f"status={rec['status']}")
    if rec["finish_reason"] != "length":
        reasons.append(f"finish_reason={rec['finish_reason']}")
    if (rec.get("usage") or {}).get("completion_tokens") != MAX_TOKENS:
        reasons.append(f"completion_tokens={(rec.get('usage') or {}).get('completion_tokens')}")
    return reasons


def size_prompt(run_id: str, words: list[str], target: int, probes: dict,
                bounds: tuple[int, int] | None = None) -> tuple[int, int | None]:
    """Smallest word count whose server-reported prompt is closest to target (1-token probes).

    `bounds` narrows the word-count search when the answer's range is known."""
    def count(n: int) -> int | None:
        if n not in probes:
            rec = stream_chat(f"{run_id}-size-{n}", INSTRUCTION + " ".join(words[:n]), 1)
            probes[n] = (rec.get("usage") or {}).get("prompt_tokens")
        return probes[n]

    # A rejected probe (None: longer than the KV pool) counts as too long.
    lo, hi = bounds or (1, min(len(words), 4 * target))
    while lo < hi:
        mid = (lo + hi) // 2
        c = count(mid)
        lo, hi = (mid + 1, hi) if c is not None and c < target else (lo, mid)
    best = min((n for n in (lo - 1, lo) if n >= 1),
               key=lambda n: abs((count(n) or 10**9) - target))
    return best, count(best)


@app.function(image=anatomy_image,
              volumes={WEIGHTS_MOUNT: weights, "/cache": jit_cache, "/results": results_vol},
              **GPU_LIMITS)
def anatomy(run_id: str) -> dict:
    """BF16 text-only server: size prompts, warm up, 5 L + 5 S untraced, traced L and S, py-spy L."""
    import hashlib
    import os
    import threading
    import time

    import httpx

    out: dict = {"server_cmd": ANATOMY_CMD, "server_env": ANATOMY_ENV, "steps": [],
                 "gpu_samples": [], "notes": [], "pyspy": {}, "traces": {}}
    t_entry = time.monotonic()
    out["container"] = {"task_id": os.environ.get("MODAL_TASK_ID"),
                        "entry_wall_ns": time.time_ns()}

    def nvsmi(fields: str) -> str:
        return subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip()

    out["gpu"] = nvsmi("name,memory.total,driver_version")
    raw = Path("/inputs/pg132_raw.txt").read_bytes()
    out["text_sha256_ok"] = hashlib.sha256(raw).hexdigest() == TEXT_SHA256
    if not out["text_sha256_ok"]:
        out["status"], out["error"] = "failed", "text hash mismatch"
        return out
    body = raw.decode("utf-8")
    words = body[body.index(TEXT_START_MARKER):].split()
    out["resolved_config"] = sh(["sgl-omni", "config", "resolve", *ANATOMY_CMD[2:4], "--text-only",
                                 "--thinker.engine.disable_radix_cache", "true",
                                 "--show", "config"])["stdout"]

    stop_sampling = threading.Event()

    def sample_gpu():
        while not stop_sampling.is_set():
            out["gpu_samples"].append([time.time_ns(), nvsmi("memory.used,utilization.gpu")])
            stop_sampling.wait(1.0)

    log_path = Path("/tmp/server.log")
    log_file = log_path.open("w", buffering=1)
    t_launch = time.monotonic()
    server = subprocess.Popen(ANATOMY_CMD, stdout=log_file, stderr=subprocess.STDOUT,
                              env={**os.environ, **ANATOMY_ENV}, start_new_session=True)
    threading.Thread(target=sample_gpu, daemon=True).start()
    trace_root = Path("/tmp/traces")

    def call(route: str, body: dict) -> None:
        resp = httpx.post(f"http://127.0.0.1:{PORT}/{route}", timeout=60, json=body)
        out["notes"].append(f"{route}: {resp.status_code} {resp.text[:300]}")

    def run(step: str, kind: str, measured: bool, **extra) -> dict:
        rec = stream_chat(f"{run_id}-{step}", prompts[kind]["text"], MAX_TOKENS)
        rec.update(step=step, kind=kind, measured=measured, invalid=valid(rec), **extra)
        rec["text_sha256"] = hashlib.sha256(rec.pop("text").encode()).hexdigest()
        out["steps"].append(rec)
        print(f"[{step}] {rec['status']} ttft={rec['ttft_s']} total={rec['completion_s']:.2f}s "
              f"usage={rec['usage']} invalid={rec['invalid']}", flush=True)
        return rec

    try:
        ready = False
        while time.monotonic() - t_launch < SERVER_READY_DEADLINE_S:
            if server.poll() is not None:
                out["notes"].append(f"server exited during startup with code {server.returncode}")
                break
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                    ready = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        out["server_ready_s"] = round(time.monotonic() - t_launch, 1) if ready else None
        out["entry_to_ready_s"] = round(time.monotonic() - t_entry, 1) if ready else None
        if not ready:
            out["status"], out["error"] = "failed", "server not healthy before deadline"
            return out
        print(f"[ready] {out['server_ready_s']} s", flush=True)
        out["gpu_after_ready"] = nvsmi("memory.used,memory.total")

        probes: dict = {}
        prompts = {}
        for kind, target in PROMPTS.items():
            n, tokens = size_prompt(run_id, words, target, probes)
            text = INSTRUCTION + " ".join(words[:n])
            prompts[kind] = {"words": n, "prompt_tokens": tokens, "target": target,
                             "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                             "ok": tokens is not None and abs(tokens - target) <= PROMPT_TOLERANCE}
        out["prompts"] = {k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in prompts.items()}
        out["size_probes"] = dict(sorted(probes.items()))
        print(f"[prompts] {out['prompts']}", flush=True)
        if not all(p["ok"] for p in prompts.values()):
            out["status"], out["error"] = "failed", "prompt sizing outside tolerance"
            return out

        # Gate: a full L request must be valid before anything else is spent.
        if run("warm_L", "L", False)["invalid"]:
            out["status"], out["error"] = "failed", "first full L request invalid"
            return out
        run("warm_S", "S", False)
        for i, kind in enumerate(MEASURE_ORDER):
            run(f"m{i:02d}_{kind}", kind, True)

        for kind in ("L", "S"):
            tdir = trace_root / kind
            call("start_profile", {"run_id": f"{run_id}-trace-{kind}", "enable_torch": True,
                                   "trace_path_template": str(tdir) + "/{stage}"})
            run(f"trace_{kind}", kind, False, torch_trace=True)
            call("stop_profile", {})
            time.sleep(30)  # stage processes export and gzip the trace

        # py-spy on every Python process in the server's session, during L's decode.
        ps = subprocess.run(["ps", "-eo", "pid,sid,args"], capture_output=True, text=True).stdout
        pids = [int(l.split()[0]) for l in ps.splitlines()[1:]
                if l.split()[1] == str(server.pid) and "python" in l]
        out["pyspy"]["processes"] = [l for l in ps.splitlines()[1:]
                                     if l.split()[1] == str(server.pid)]
        spies = {}
        result_holder: dict = {}
        t = threading.Thread(target=lambda: result_holder.setdefault(
            "rec", run("pyspy_L", "L", False, pyspy=True)))
        t.start()
        time.sleep(2.0)  # past the 8k prefill, into decode
        for pid in pids:
            spies[pid] = subprocess.Popen(
                [f"/opt/sglang/bin/py-spy", "record", "--pid", str(pid), "--duration", "4",
                 "--rate", "200", "--native", "--format", "raw", "--output", f"/tmp/pyspy-{pid}.txt"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for pid, proc in spies.items():
            try:
                log = proc.communicate(timeout=60)[0]
            except subprocess.TimeoutExpired:
                proc.kill()
                log = "timeout"
            f = Path(f"/tmp/pyspy-{pid}.txt")
            out["pyspy"][str(pid)] = {"returncode": proc.returncode, "log": (log or "")[-2000:],
                                      "collapsed": f.read_text() if f.exists() else None}
        t.join(timeout=REQUEST_DEADLINE_S)

        out["gpu_after_requests"] = nvsmi("memory.used,memory.total")
        out["status"] = "completed"
    except Exception as exc:
        out["status"], out["error"] = "failed", repr(exc)[:2000]
    finally:
        stop_sampling.set()
        if server.poll() is None:
            os.killpg(server.pid, 15)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, 9)
        out["server_returncode"] = server.returncode
        log_file.close()
        log = log_path.read_text(errors="replace")
        out["server_log_bytes"] = len(log)
        out["server_log"] = log[-4_000_000:]
        dest = Path("/results/1_anatomy") / run_id
        dest.mkdir(parents=True, exist_ok=True)
        for path in sorted(trace_root.rglob("*")) if trace_root.exists() else []:
            if path.is_file():
                rel = str(path.relative_to(trace_root))
                target = dest / "traces" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
                out["traces"][rel] = path.stat().st_size
        (dest / "run.json").write_text(json.dumps(out, indent=2, default=str))
        for vol in (results_vol, jit_cache):
            try:
                vol.commit()
            except Exception as exc:
                out["notes"].append(f"volume commit failed: {exc!r}")
    return out


# ---- Experiment 2b: why is FP8 slower at batch 1? Backend sessions ----------

# (session, model dir, extra flags, traced); most important first, A last as drift check.
BACKEND_SESSIONS = (
    ("B_default", FP8_DIR, [], True),
    ("B_deepgemm", FP8_DIR, ["--thinker.engine.fp8_gemm_runner_backend", "deep_gemm"], True),
    ("B_moe_triton", FP8_DIR, ["--thinker.engine.moe_runner_backend", "triton"], True),
    ("A", BF16_DIR, [], False),
)
BACKEND_REPEATS = 5
SESSION_BUDGET_S = 24 * 60  # start no new session after this; the run timeout is 30 min


@app.function(image=anatomy_image,
              volumes={WEIGHTS_MOUNT: weights, "/cache": jit_cache, "/results": results_vol},
              **GPU_LIMITS)
def fp8_backends(run_id: str) -> dict:
    """Per session: server start, 1 warmup + 5 untraced 8k/1k requests, optionally 1 traced."""
    import hashlib
    import os
    import re as _re
    import time

    import httpx

    t0 = time.monotonic()
    out: dict = {"sessions": {}, "notes": [], "traces": {}}
    out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    out["backend_choices"] = sh([SGLANG_PY, "-c", (
        "import json, sglang.srt.server_args as sa\n"
        "print(json.dumps({k: list(getattr(sa, k)) for k in dir(sa) if k.endswith('BACKEND_CHOICES')}))")])
    raw = Path("/inputs/pg132_raw.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEXT_SHA256:
        out["status"], out["error"] = "failed", "text hash mismatch"
        return out
    body_text = raw.decode("utf-8")
    words = body_text[body_text.index(TEXT_START_MARKER):].split()
    long_text = INSTRUCTION + " ".join(words[:6057])  # experiment 1's L prompt (8,000 tokens)
    dest = Path("/results/2b_fp8_backends") / run_id
    trace_root = Path("/tmp/traces")

    for name, model_dir, extra, traced in BACKEND_SESSIONS:
        if time.monotonic() - t0 > SESSION_BUDGET_S:
            out["sessions"][name] = {"status": "skipped", "error": "session time budget reached"}
            continue
        cmd = server_cmd(model_dir, extra)
        sess = {"server_cmd": cmd, "requests": []}
        out["sessions"][name] = sess
        log_path = Path(f"/tmp/server-{name}.log")
        log_file = log_path.open("w", buffering=1)
        t_launch = time.monotonic()
        server = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                  env={**os.environ, **ANATOMY_ENV}, start_new_session=True)
        try:
            ready = False
            while time.monotonic() - t_launch < SERVER_READY_DEADLINE_S:
                if server.poll() is not None:
                    sess["error"] = f"server exited during startup with code {server.returncode}"
                    break
                try:
                    if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                        ready = True
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            sess["server_ready_s"] = round(time.monotonic() - t_launch, 1) if ready else None
            print(f"[{name}] ready={ready} {sess['server_ready_s']} s", flush=True)
            if not ready:
                sess["status"] = "failed"
                continue
            for i in range(1 + BACKEND_REPEATS):
                rec = stream_chat(f"{run_id}-{name}-{i}", long_text, MAX_TOKENS)
                rec["invalid"] = valid(rec)
                rec["measured"] = i > 0
                rec["text_sha256"] = hashlib.sha256(rec.pop("text").encode()).hexdigest()
                if rec["ttft_s"] is not None and rec["usage"]:
                    rec["decode_ms_per_tok"] = ((rec["completion_s"] - rec["ttft_s"]) * 1000
                                                / max(1, rec["usage"]["completion_tokens"] - 1))
                rec.pop("chunk_offsets_us", None)
                sess["requests"].append(rec)
                print(f"[{name}/{i}] decode={rec.get('decode_ms_per_tok')} invalid={rec['invalid']}", flush=True)
            if traced:
                tdir = trace_root / name
                r = httpx.post(f"http://127.0.0.1:{PORT}/start_profile", timeout=60, json={
                    "run_id": f"{run_id}-{name}", "enable_torch": True,
                    "trace_path_template": str(tdir) + "/{stage}"})
                sess["profile_start"] = f"{r.status_code} {r.text[:200]}"
                rec = stream_chat(f"{run_id}-{name}-trace", long_text, MAX_TOKENS)
                httpx.post(f"http://127.0.0.1:{PORT}/stop_profile", timeout=60, json={})
                time.sleep(30)  # trace export and gzip
                rec["invalid"] = valid(rec)
                rec.pop("text", None)
                rec.pop("chunk_offsets_us", None)
                sess["trace_request"] = rec
            sess["status"] = "completed"
        except Exception as exc:
            sess["status"], sess["error"] = "failed", repr(exc)[:2000]
        finally:
            if server.poll() is None:
                os.killpg(server.pid, 15)
                try:
                    server.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, 9)
                    server.wait(timeout=30)
            log_file.close()
            log = log_path.read_text(errors="replace")
            sess["server_log"] = log[-3_000_000:]
            sess["log_facts"] = [l[:400] for l in log.splitlines() if _re.search(
                r"KV Cache is allocated|Load weight end|backend policy|Config file not found|"
                r"Using default|Traceback|Error", l)][:200]
            # Persist this session now, so a later timeout cannot lose it.
            dest.mkdir(parents=True, exist_ok=True)
            for path in sorted((trace_root / name).rglob("*")) if (trace_root / name).exists() else []:
                if path.is_file():
                    rel = str(path.relative_to(trace_root))
                    target = dest / "traces" / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
                    out["traces"][rel] = path.stat().st_size
            (dest / "run.json").write_text(json.dumps(out, indent=2, default=str))
            results_vol.commit()
            for _ in range(60):
                used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True).stdout.strip()
                if used.isdigit() and int(used) < 2000:
                    break
                time.sleep(2)
    out["status"] = "completed"
    (dest / "run.json").write_text(json.dumps(out, indent=2, default=str))
    results_vol.commit()
    jit_cache.commit()
    return out


# ---- Experiment 2b-2: combined FP8 backends, quality set, 8-concurrent check --

COMBINED_FLAGS = ["--thinker.engine.fp8_gemm_runner_backend", "deep_gemm",
                  "--thinker.engine.moe_runner_backend", "triton"]
NEEDLE_SEED = 1234
NEEDLE_CONTEXT_WORDS = (1500, 3000, 6000)  # about 2k, 4k, 8k tokens (8,000 / 6,057 tokens per word)
NEEDLES_PER_CONTEXT = 20
CONCURRENCY = 8


def build_needles(words: list[str]) -> list[dict]:
    """Deterministic needle-retrieval items: one code sentence hidden in a text window."""
    import random

    rng = random.Random(NEEDLE_SEED)
    items = []
    for ctx in NEEDLE_CONTEXT_WORDS:
        for k in range(NEEDLES_PER_CONTEXT):
            start = rng.randrange(0, len(words) - ctx)
            hay = words[start:start + ctx]
            locker, code = rng.randrange(100, 1000), rng.randrange(100000, 1000000)
            depth = rng.random()
            at = int(depth * len(hay))
            needle = f"(Note: the access code for locker {locker} is {code}.)"
            text = " ".join(hay[:at] + [needle] + hay[at:])
            prompt = (f"Read the following text.\n\n{text}\n\nQuestion: What is the access code "
                      f"for locker {locker}? Answer with the number only.")
            items.append({"id": f"w{ctx}_{k:02d}", "context_words": ctx, "depth": round(depth, 3),
                          "answer": str(code), "prompt": prompt})
    return items


def wait_for_trace(tdir: Path, timeout_s: int = 180) -> bool:
    """Wait until the exporter's background gzip has produced a .gz file."""
    import time

    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if any(p.suffix == ".gz" for p in tdir.rglob("*")) if tdir.exists() else False:
            time.sleep(5)
            return True
        time.sleep(5)
    return False


COMBINED_SESSIONS = (  # (name, model dir, flags, plan)
    ("B_combined", FP8_DIR, COMBINED_FLAGS, ("measure", "trace", "quality", "concurrent")),
    ("B_default", FP8_DIR, [], ("concurrent",)),
    ("A", BF16_DIR, [], ("quality",)),
)


@app.function(image=anatomy_image,
              volumes={WEIGHTS_MOUNT: weights, "/cache": jit_cache, "/results": results_vol},
              **GPU_LIMITS)
def fp8_combined(run_id: str) -> dict:
    import hashlib
    import os
    import re as _re
    import time
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    t0 = time.monotonic()
    out: dict = {"sessions": {}, "traces": {}, "notes": []}
    out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    raw = Path("/inputs/pg132_raw.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEXT_SHA256:
        out["status"], out["error"] = "failed", "text hash mismatch"
        return out
    body_text = raw.decode("utf-8")
    words = body_text[body_text.index(TEXT_START_MARKER):].split()
    long_text = INSTRUCTION + " ".join(words[:6057])
    needles = build_needles(words)
    out["needles"] = [{k: v for k, v in n.items() if k != "prompt"} | {
        "prompt_sha256": hashlib.sha256(n["prompt"].encode()).hexdigest()} for n in needles]
    dest = Path("/results/2b2_fp8_combined") / run_id
    trace_root = Path("/tmp/traces")

    def timed_request(tag: str) -> dict:
        rec = stream_chat(tag, long_text, MAX_TOKENS)
        rec["invalid"] = valid(rec)
        rec["text_sha256"] = hashlib.sha256(rec.pop("text").encode()).hexdigest()
        if rec["ttft_s"] is not None and rec["usage"]:
            rec["decode_ms_per_tok"] = ((rec["completion_s"] - rec["ttft_s"]) * 1000
                                        / max(1, rec["usage"]["completion_tokens"] - 1))
        rec.pop("chunk_offsets_us", None)
        return rec

    for name, model_dir, flags, plan in COMBINED_SESSIONS:
        if time.monotonic() - t0 > SESSION_BUDGET_S:
            out["sessions"][name] = {"status": "skipped", "error": "session time budget reached"}
            continue
        cmd = server_cmd(model_dir, flags)
        sess = {"server_cmd": cmd, "plan": plan, "requests": []}
        out["sessions"][name] = sess
        log_path = Path(f"/tmp/server-{name}.log")
        log_file = log_path.open("w", buffering=1)
        t_launch = time.monotonic()
        server = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                  env={**os.environ, **ANATOMY_ENV}, start_new_session=True)
        try:
            ready = False
            while time.monotonic() - t_launch < SERVER_READY_DEADLINE_S:
                if server.poll() is not None:
                    sess["error"] = f"server exited during startup with code {server.returncode}"
                    break
                try:
                    if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                        ready = True
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            sess["server_ready_s"] = round(time.monotonic() - t_launch, 1) if ready else None
            print(f"[{name}] ready={ready} {sess['server_ready_s']} s", flush=True)
            if not ready:
                sess["status"] = "failed"
                continue
            warm = timed_request(f"{run_id}-{name}-warm")
            warm["measured"] = False
            sess["requests"].append(warm)
            if "measure" in plan:
                for i in range(BACKEND_REPEATS):
                    rec = timed_request(f"{run_id}-{name}-{i}")
                    rec["measured"] = True
                    sess["requests"].append(rec)
                    print(f"[{name}/{i}] decode={rec.get('decode_ms_per_tok')} invalid={rec['invalid']}", flush=True)
            if "trace" in plan:
                tdir = trace_root / name
                httpx.post(f"http://127.0.0.1:{PORT}/start_profile", timeout=60, json={
                    "run_id": f"{run_id}-{name}", "enable_torch": True,
                    "trace_path_template": str(tdir) + "/{stage}"})
                rec = timed_request(f"{run_id}-{name}-trace")
                httpx.post(f"http://127.0.0.1:{PORT}/stop_profile", timeout=60, json={})
                rec["trace_gz_ready"] = wait_for_trace(tdir)
                sess["trace_request"] = rec
            if "quality" in plan:
                q = []
                for item in needles:
                    t = time.monotonic()
                    g = generate(item["prompt"], 16)
                    q.append({"id": item["id"], "context_words": item["context_words"],
                              "status": g["status"], "text": g["text"], "nonfinite": g["nonfinite_logprobs"],
                              "correct": item["answer"] in (g["text"] or ""),
                              "latency_s": round(time.monotonic() - t, 4)})
                sess["needles"] = q
                sess["short"] = [generate(p, SHORT_TOKENS) for p in SHORT_PROMPTS]
                print(f"[{name}] needles correct {sum(x['correct'] for x in q)}/{len(q)}", flush=True)
            if "concurrent" in plan:
                t = time.monotonic()
                with ThreadPoolExecutor(CONCURRENCY) as pool:
                    recs = list(pool.map(lambda i: timed_request(f"{run_id}-{name}-c{i}"), range(CONCURRENCY)))
                wall = time.monotonic() - t
                toks = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in recs)
                sess["concurrent"] = {"n": CONCURRENCY, "wall_s": round(wall, 3), "output_tokens": toks,
                                      "output_tok_per_s": round(toks / wall, 1), "requests": recs}
                print(f"[{name}] c{CONCURRENCY}: {toks} tok in {wall:.2f} s = {toks / wall:.0f} tok/s "
                      f"invalid={[r['invalid'] for r in recs if r['invalid']]}", flush=True)
            sess["status"] = "completed"
        except Exception as exc:
            sess["status"], sess["error"] = "failed", repr(exc)[:2000]
        finally:
            if server.poll() is None:
                os.killpg(server.pid, 15)
                try:
                    server.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, 9)
                    server.wait(timeout=30)
            log_file.close()
            log = log_path.read_text(errors="replace")
            sess["server_log"] = log[-3_000_000:]
            sess["log_facts"] = [l[:400] for l in log.splitlines() if _re.search(
                r"KV Cache is allocated|Load weight end|backend policy|Config file not found|"
                r"Using default|DeepGEMM|deep_gemm|Traceback|Error", l)][:200]
            dest.mkdir(parents=True, exist_ok=True)
            for path in sorted((trace_root / name).rglob("*.gz")) if (trace_root / name).exists() else []:
                rel = str(path.relative_to(trace_root))
                target = dest / "traces" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
                out["traces"][rel] = path.stat().st_size
            (dest / "run.json").write_text(json.dumps(out, indent=2, default=str))
            results_vol.commit()
            for _ in range(60):
                used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True).stdout.strip()
                if used.isdigit() and int(used) < 2000:
                    break
                time.sleep(2)
    out["status"] = "completed"
    (dest / "run.json").write_text(json.dumps(out, indent=2, default=str))
    results_vol.commit()
    jit_cache.commit()
    return out


# ---- Experiment 4a: memory fraction and closed-loop concurrency scan ----------

# (name, model dir, extra flags, mem_fraction_static, concurrency levels, condition)
CAPACITY_SESSIONS = (
    ("A_0.784", BF16_DIR, [], "0.784", (1, 2, 4, 8), None),
    ("A_0.90", BF16_DIR, [], "0.90", (1, 2, 4, 8, 12, 16), None),
    ("FP8c_0.90", FP8_DIR, COMBINED_FLAGS, "0.90", (1, 2, 4, 8, 16, 24, 32), None),
    ("A_0.93", BF16_DIR, [], "0.93", (1, 2, 4, 8, 12, 16, 24), "A_0.90 left >= 4 GiB unused"),
)
CAPACITY_WINDOW_STEP = 500  # words between prompt start offsets
CAPACITY_GATE_PEAK = 0.95   # peak used / total GPU memory


@app.function(image=anatomy_image,
              volumes={WEIGHTS_MOUNT: weights, "/cache": jit_cache, "/results": results_vol},
              **GPU_LIMITS)
def capacity_scan(run_id: str) -> dict:
    import hashlib
    import os
    import re as _re
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    t0 = time.monotonic()
    out: dict = {"sessions": {}, "notes": []}
    out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    total_mib = int(out["gpu"].split(",")[1])
    raw = Path("/inputs/pg132_raw.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEXT_SHA256:
        out["status"], out["error"] = "failed", "text hash mismatch"
        return out
    body_text = raw.decode("utf-8")
    words = body_text[body_text.index(TEXT_START_MARKER):].split()
    # 32 distinct 8k-class prompts (6,057 words each, start offsets 500 words apart).
    prompts = [INSTRUCTION + " ".join(words[i * CAPACITY_WINDOW_STEP: i * CAPACITY_WINDOW_STEP + 6057])
               for i in range(32)]
    out["prompt_sha256"] = [hashlib.sha256(p.encode()).hexdigest() for p in prompts]
    dest = Path("/results/4a_capacity_scan") / run_id

    def used_mib() -> int | None:
        v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip()
        return int(v) if v.isdigit() else None

    def one(tag: str, prompt: str) -> dict:
        rec = stream_chat(tag, prompt, MAX_TOKENS)
        rec["invalid"] = valid(rec)
        rec.pop("text", None)
        t = rec.pop("chunk_offsets_us", [])
        gaps = sorted((b - a) / 1000 for a, b in zip(t, t[1:]))
        rec["tbt_ms_p50"] = gaps[len(gaps) // 2] if gaps else None
        rec["tbt_ms_p95"] = gaps[int(len(gaps) * 0.95)] if gaps else None
        if rec["ttft_s"] is not None and rec["usage"]:
            rec["decode_ms_per_tok"] = ((rec["completion_s"] - rec["ttft_s"]) * 1000
                                        / max(1, rec["usage"]["completion_tokens"] - 1))
        return rec

    for name, model_dir, extra, frac, levels, condition in CAPACITY_SESSIONS:
        if condition and not out.get("allow_0.93"):
            out["sessions"][name] = {"status": "skipped", "error": f"condition not met: {condition}"}
            continue
        if time.monotonic() - t0 > SESSION_BUDGET_S:
            out["sessions"][name] = {"status": "skipped", "error": "session time budget reached"}
            continue
        cmd = ["sgl-omni", "serve", "--model-path", model_dir, "--text-only",
               "--host", "127.0.0.1", "--port", str(PORT), "--model-name", SERVED_MODEL,
               "--thinker.engine.disable_radix_cache", "true",
               "--thinker.engine.mem_fraction_static", frac, *extra]
        sess = {"server_cmd": cmd, "mem_fraction_static": frac, "levels": {}}
        out["sessions"][name] = sess
        log_path = Path(f"/tmp/server-{name}.log")
        log_file = log_path.open("w", buffering=1)
        t_launch = time.monotonic()
        server = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                  env={**os.environ, **ANATOMY_ENV}, start_new_session=True)
        peak = {"mib": 0}
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                u = used_mib()
                if u:
                    peak["mib"] = max(peak["mib"], u)
                stop.wait(0.5)

        threading.Thread(target=sampler, daemon=True).start()
        try:
            ready = False
            while time.monotonic() - t_launch < SERVER_READY_DEADLINE_S:
                if server.poll() is not None:
                    sess["error"] = f"server exited during startup with code {server.returncode}"
                    break
                try:
                    if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                        ready = True
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            sess["server_ready_s"] = round(time.monotonic() - t_launch, 1) if ready else None
            sess["used_after_ready_mib"] = used_mib()
            print(f"[{name}] ready={ready} {sess['server_ready_s']} s used={sess['used_after_ready_mib']} MiB", flush=True)
            if not ready:
                sess["status"] = "failed"
                continue
            one(f"{run_id}-{name}-warm", prompts[0])
            for n in levels:
                peak["mib"] = 0
                t = time.monotonic()
                with ThreadPoolExecutor(n) as pool:
                    recs = list(pool.map(lambda i: one(f"{run_id}-{name}-c{n}-{i}", prompts[i % len(prompts)]),
                                         range(n)))
                wall = time.monotonic() - t
                toks = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in recs)
                lvl = {"n": n, "wall_s": round(wall, 3), "output_tokens": toks,
                       "output_tok_per_s": round(toks / wall, 1), "peak_used_mib": peak["mib"],
                       "invalid": [r["invalid"] for r in recs if r["invalid"]],
                       "server_alive": server.poll() is None, "requests": recs}
                sess["levels"][str(n)] = lvl
                dm = sorted(r.get("decode_ms_per_tok") or 0 for r in recs)
                tt = sorted(r.get("ttft_s") or 0 for r in recs)
                print(f"[{name} c{n}] {lvl['output_tok_per_s']} tok/s wall={wall:.1f}s decode p50={dm[len(dm)//2]:.2f} ms "
                      f"ttft max={tt[-1]:.2f}s peak={peak['mib']} MiB invalid={len(lvl['invalid'])}", flush=True)
                if lvl["invalid"] or not lvl["server_alive"]:
                    sess["stopped_at"] = n
                    break
            all_peak = max((l["peak_used_mib"] for l in sess["levels"].values()), default=None)
            sess["peak_used_mib"] = all_peak
            sess["gate_pass"] = (all(not l["invalid"] and l["server_alive"] for l in sess["levels"].values())
                                 and all_peak is not None and all_peak <= CAPACITY_GATE_PEAK * total_mib)
            if name == "A_0.90":
                out["allow_0.93"] = bool(sess["gate_pass"] and (total_mib - all_peak) >= 4096)
            sess["status"] = "completed"
        except Exception as exc:
            sess["status"], sess["error"] = "failed", repr(exc)[:2000]
        finally:
            stop.set()
            if server.poll() is None:
                os.killpg(server.pid, 15)
                try:
                    server.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, 9)
                    server.wait(timeout=30)
            log_file.close()
            log = log_path.read_text(errors="replace")
            sess["server_log"] = log[-3_000_000:]
            sess["log_facts"] = [l[:400] for l in log.splitlines() if _re.search(
                r"KV Cache is allocated|Load weight end|backend policy|max_running_requests|"
                r"out of memory|OutOfMemory|Traceback|Error|Rejecting", l)][:200]
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "run.json").write_text(json.dumps(out, indent=2, default=str))
            results_vol.commit()
            for _ in range(60):
                u = used_mib()
                if u is not None and u < 2000:
                    break
                time.sleep(2)
    out["status"] = "completed"
    (dest / "run.json").write_text(json.dumps(out, indent=2, default=str))
    results_vol.commit()
    jit_cache.commit()
    return out


# ---- Experiment 4b: open-loop capacity sweep against the frozen criteria ------

# Frozen in 4_capacity/README.md before any run; the req/s values follow from 4a.
SWEEP_FRACTIONS = (0.25, 0.5, 0.7, 0.85, 1.0, 1.1)
SWEEP_WARMUP_S, SWEEP_WINDOW_S = 60, 300
SWEEP_MIN_SAMPLE = 60
SWEEP_QUEUE_BOUND = 64
SWEEP_REQUEST_DEADLINE_S = 120
SWEEP_TTFT_P95_S, SWEEP_TBT_P95_MS = 3.0, 60.0
SWEEP_PROMPTS, SWEEP_TARGET_TOKENS = 50, 8000
SWEEP_DRAIN_S = 130  # after the last arrival: the request deadline plus slack
SWEEP_RUN_S = 80 * 60  # budget.py row 4b: 10 + 80 + 2 min
SWEEP_LIMITS = dict(GPU_LIMITS, timeout=SWEEP_RUN_S)
# (name, model dir, flags, fraction, 4a closed-loop max tok/s, fractions run, extra closed-loop levels)
# Fractions whose expected sample in 300 s is below 60 (0.25 x 0.73 and 0.5 x 0.32 req/s) are not run:
# the frozen rule would report them as under-sampled and unused (dated note, 2026-10-30).
SWEEP_SESSIONS = (
    ("FP8c_0.90", FP8_DIR, COMBINED_FLAGS, "0.90", 1390.0, SWEEP_FRACTIONS, (32, 40)),
    ("A_0.90", BF16_DIR, [], "0.90", 730.0, SWEEP_FRACTIONS[1:], ()),
    ("A_0.784", BF16_DIR, [], "0.784", 318.0, (0.7, 1.0), ()),
)


def pct(xs: list, q: float):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def rate_summary(requests: list[dict], queue_bound_hit: bool, server_alive: bool) -> dict:
    """Apply the frozen 4b criteria to one rate's request records."""
    lo, hi = SWEEP_WARMUP_S, SWEEP_WARMUP_S + SWEEP_WINDOW_S
    m = [r for r in requests if lo <= r["send_offset_s"] < hi]
    done = [r for r in m if not r["invalid"]]
    ends = [r["send_offset_s"] + r["completion_s"] for r in requests if not r["invalid"]]
    gaps = []
    for r in done:
        c = r["chunk_offsets_us"]
        gaps += [(b - a) / 1000 for a, b in zip(c, c[1:])]
    s = {"sent": len(m), "completed": len(done), "invalid": sum(bool(r["invalid"]) for r in m),
         "offered_rps": round(len(m) / SWEEP_WINDOW_S, 4),
         "completed_rps": round(sum(lo <= e < hi for e in ends) / SWEEP_WINDOW_S, 4),
         "ttft_s": {q: pct([r["ttft_s"] for r in done], p) for q, p in (("p50", .5), ("p95", .95), ("p99", .99))},
         "tbt_ms": {q: pct(gaps, p) for q, p in (("p50", .5), ("p95", .95), ("p99", .99))},
         "output_tok_per_s": round(sum((r.get("usage") or {}).get("completion_tokens", 0)
                                       for r in done) / SWEEP_WINDOW_S, 1)}
    reasons = []
    if s["invalid"]:
        reasons.append(f"{s['invalid']} invalid requests")
    if queue_bound_hit:
        reasons.append("client in-flight reached the queue bound")
    if not server_alive:
        reasons.append("server exited")
    if s["completed_rps"] < 0.95 * s["offered_rps"]:
        reasons.append(f"completed {s['completed_rps']} < 95% of offered {s['offered_rps']} req/s")
    if s["ttft_s"]["p95"] is None or s["ttft_s"]["p95"] > SWEEP_TTFT_P95_S:
        reasons.append(f"p95 TTFT {s['ttft_s']['p95']} s")
    if s["tbt_ms"]["p95"] is None or s["tbt_ms"]["p95"] > SWEEP_TBT_P95_MS:
        reasons.append(f"p95 TBT {s['tbt_ms']['p95']} ms")
    s["under_sampled"] = s["completed"] < SWEEP_MIN_SAMPLE
    s["fail_reasons"] = reasons
    s["verdict"] = "fail" if reasons else ("under-sampled" if s["under_sampled"] else "pass")
    return s


@app.function(image=anatomy_image,
              volumes={WEIGHTS_MOUNT: weights, "/cache": jit_cache, "/results": results_vol},
              **SWEEP_LIMITS)
def capacity_sweep(run_id: str) -> dict:
    """Per session: size 50 prompts, optional closed-loop levels, then Poisson rates until one fails."""
    import hashlib
    import os
    import random
    import re as _re
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    t0 = time.monotonic()
    out: dict = {"sessions": {}, "notes": [], "criteria": {
        "fractions": SWEEP_FRACTIONS, "warmup_s": SWEEP_WARMUP_S, "window_s": SWEEP_WINDOW_S,
        "min_sample": SWEEP_MIN_SAMPLE, "queue_bound": SWEEP_QUEUE_BOUND,
        "request_deadline_s": SWEEP_REQUEST_DEADLINE_S, "ttft_p95_s": SWEEP_TTFT_P95_S,
        "tbt_p95_ms": SWEEP_TBT_P95_MS,
        "offered_rate": "requests sent in the measured window / window",
        "completed_rate": "requests completed in the measured window / window"}}
    out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    raw = Path("/inputs/pg132_raw.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEXT_SHA256:
        out["status"], out["error"] = "failed", "text hash mismatch"
        return out
    body_text = raw.decode("utf-8")
    words = body_text[body_text.index(TEXT_START_MARKER):].split()
    dest = Path("/results/4b_capacity_sweep") / run_id
    prompts: list[str] = []

    def left() -> float:
        return SWEEP_RUN_S - (time.monotonic() - t0)

    def used_mib() -> int | None:
        v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip()
        return int(v) if v.isdigit() else None

    def persist() -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "run.json").write_text(json.dumps(out, default=str))
        results_vol.commit()

    def one(tag: str, prompt: str, offset_s: float | None = None) -> dict:
        rec = stream_chat(tag, prompt, MAX_TOKENS, deadline_s=SWEEP_REQUEST_DEADLINE_S)
        rec["invalid"] = valid(rec)
        rec["text_sha256"] = hashlib.sha256(rec.pop("text").encode()).hexdigest()
        rec["send_offset_s"] = offset_s
        return rec

    for si, (name, model_dir, flags, frac, max_tok_s, fractions, closed_levels) in enumerate(SWEEP_SESSIONS):
        if left() < 10 * 60:
            out["sessions"][name] = {"status": "skipped", "error": "less than 10 min of run time left"}
            continue
        cmd = ["sgl-omni", "serve", "--model-path", model_dir, "--text-only",
               "--host", "127.0.0.1", "--port", str(PORT), "--model-name", SERVED_MODEL,
               "--thinker.engine.disable_radix_cache", "true",
               "--thinker.engine.mem_fraction_static", frac, *flags]
        sess = {"server_cmd": cmd, "mem_fraction_static": frac, "closed_loop": {}, "rates": []}
        out["sessions"][name] = sess
        log_path = Path(f"/tmp/server-{name}.log")
        log_file = log_path.open("w", buffering=1)
        t_launch = time.monotonic()
        server = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                  env={**os.environ, **ANATOMY_ENV}, start_new_session=True)
        peak = {"mib": 0}
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                u = used_mib()
                if u:
                    peak["mib"] = max(peak["mib"], u)
                stop.wait(0.5)

        threading.Thread(target=sampler, daemon=True).start()
        try:
            ready = False
            while time.monotonic() - t_launch < SERVER_READY_DEADLINE_S:
                if server.poll() is not None:
                    sess["error"] = f"server exited during startup with code {server.returncode}"
                    break
                try:
                    if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                        ready = True
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            sess["server_ready_s"] = round(time.monotonic() - t_launch, 1) if ready else None
            print(f"[{name}] ready={ready} {sess['server_ready_s']} s", flush=True)
            if not ready:
                sess["status"] = "failed"
                continue

            # Prompts: sized once with the server's own count (same tokenizer for every session),
            # then every session sends each once with max_tokens 1, which checks the counts and
            # warms per-shape compiles before any measurement.
            if not prompts:
                sizing = []
                for i in range(SWEEP_PROMPTS):
                    window = words[i * CAPACITY_WINDOW_STEP:]
                    n, tokens = size_prompt(f"{run_id}-p{i}", window, SWEEP_TARGET_TOKENS, {},
                                            bounds=(5600, 6500))
                    prompts.append(INSTRUCTION + " ".join(window[:n]))
                    sizing.append({"index": i, "start_word": i * CAPACITY_WINDOW_STEP, "words": n,
                                   "prompt_tokens": tokens,
                                   "sha256": hashlib.sha256(prompts[-1].encode()).hexdigest()})
                out["prompts"] = sizing
                out["prompts_ok"] = all(p["prompt_tokens"] is not None and
                                        abs(p["prompt_tokens"] - SWEEP_TARGET_TOKENS) <= PROMPT_TOLERANCE
                                        for p in sizing)
                print(f"[prompts] ok={out['prompts_ok']} tokens "
                      f"{min(p['prompt_tokens'] or 0 for p in sizing)}-{max(p['prompt_tokens'] or 0 for p in sizing)}",
                      flush=True)
            counts = [(stream_chat(f"{run_id}-{name}-check{i}", p, 1).get("usage") or {}).get("prompt_tokens")
                      for i, p in enumerate(prompts)]
            sess["prompt_tokens"] = counts
            sess["prompts_match"] = counts == [p["prompt_tokens"] for p in out["prompts"]]
            one(f"{run_id}-{name}-warm", prompts[0])

            for n in closed_levels:
                peak["mib"] = 0
                t = time.monotonic()
                with ThreadPoolExecutor(n) as pool:
                    recs = list(pool.map(lambda i: one(f"{run_id}-{name}-c{n}-{i}", prompts[i % len(prompts)]),
                                         range(n)))
                wall = time.monotonic() - t
                toks = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in recs)
                for r in recs:
                    r.pop("chunk_offsets_us", None)
                sess["closed_loop"][str(n)] = {"n": n, "wall_s": round(wall, 3), "output_tokens": toks,
                                               "output_tok_per_s": round(toks / wall, 1), "peak_used_mib": peak["mib"],
                                               "invalid": [r["invalid"] for r in recs if r["invalid"]],
                                               "ttft_max_s": max((r["ttft_s"] or 0) for r in recs), "requests": recs}
                print(f"[{name} closed c{n}] {toks / wall:.0f} tok/s invalid={sum(bool(r['invalid']) for r in recs)}",
                      flush=True)
            valid_levels = [l["output_tok_per_s"] for l in sess["closed_loop"].values() if not l["invalid"]]
            sess["max_tok_s_4a"] = max_tok_s
            sess["max_tok_s"] = max([max_tok_s, *valid_levels])
            base_rps = sess["max_tok_s"] / MAX_TOKENS
            sess["fractions"] = list(fractions)
            persist()

            for ri, f in enumerate(fractions):
                if left() < SWEEP_WARMUP_S + SWEEP_WINDOW_S + SWEEP_DRAIN_S + 30:
                    sess["rates"].append({"fraction": f, "status": "skipped", "error": "run time left too short"})
                    break
                lam = f * base_rps
                rng = random.Random(4000 + 10 * si + ri)  # fixed seed per (session, rate)
                arrivals, t_a = [], rng.expovariate(lam)
                while t_a < SWEEP_WARMUP_S + SWEEP_WINDOW_S:
                    arrivals.append(t_a)
                    t_a += rng.expovariate(lam)
                rate = {"fraction": f, "offered_rps_target": round(lam, 4), "seed": 4000 + 10 * si + ri,
                        "arrivals": len(arrivals), "requests": [], "queue_bound_hit": False,
                        "max_in_flight": 0}
                sess["rates"].append(rate)
                inflight = {"n": 0}
                lock = threading.Lock()
                peak["mib"] = 0

                def task(i: int, off: float) -> dict:
                    try:
                        return one(f"{run_id}-{name}-r{ri}-{i}", prompts[i % len(prompts)], off)
                    finally:
                        with lock:
                            inflight["n"] -= 1

                t_start = time.monotonic()
                rate["start_wall_ns"] = time.time_ns()
                futures = []
                with ThreadPoolExecutor(SWEEP_QUEUE_BOUND + 8) as pool:
                    for i, off in enumerate(arrivals):
                        delay = t_start + off - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                        with lock:
                            if inflight["n"] >= SWEEP_QUEUE_BOUND:
                                rate["queue_bound_hit"] = True
                                break
                            inflight["n"] += 1
                            rate["max_in_flight"] = max(rate["max_in_flight"], inflight["n"])
                        futures.append(pool.submit(task, i, round(time.monotonic() - t_start, 4)))
                    for fut in futures:
                        rate["requests"].append(fut.result(timeout=SWEEP_DRAIN_S + SWEEP_REQUEST_DEADLINE_S))
                rate["peak_used_mib"] = peak["mib"]
                rate["server_alive"] = server.poll() is None

                s = rate_summary(rate["requests"], rate["queue_bound_hit"], rate["server_alive"])
                rate["summary"] = s
                print(f"[{name} {f}x = {lam:.3f} req/s] {s['verdict']} sent={s['sent']} done={s['completed']} "
                      f"ttft p95={s['ttft_s']['p95']} tbt p95={s['tbt_ms']['p95']} reasons={s['fail_reasons']}", flush=True)
                persist()
                if s["verdict"] == "fail":
                    break
            sess["status"] = "completed"
        except Exception as exc:
            sess["status"], sess["error"] = "failed", repr(exc)[:2000]
        finally:
            stop.set()
            if server.poll() is None:
                os.killpg(server.pid, 15)
                try:
                    server.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, 9)
                    server.wait(timeout=30)
            log_file.close()
            log = log_path.read_text(errors="replace")
            # Engine running/queued counts come from SGLang's batch log lines (container clock).
            sess["batch_log"] = [l[:300] for l in log.splitlines()
                                 if _re.search(r"(Decode|Prefill) batch", l)][-200_000:]
            sess["server_log"] = log[-2_000_000:]
            sess["log_facts"] = [l[:400] for l in log.splitlines() if _re.search(
                r"KV Cache is allocated|Load weight end|backend policy|max_running_requests|"
                r"out of memory|OutOfMemory|Traceback|Error|Rejecting|retract", l)][:300]
            persist()
            for _ in range(60):
                u = used_mib()
                if u is not None and u < 2000:
                    break
                time.sleep(2)
    out["status"] = "completed"
    persist()
    jit_cache.commit()
    return out


# ---- Experiment 2c: tuned Triton block-FP8 dense configs for the thinker ------

# The two dense FP8 shapes the 2b server logs report without a tuned config.
TUNE_SHAPES = ((5120, 2048), (2048, 4096))  # (N, K): QKV projection, O projection
# Decode sizes only (batch 1 to c64). The first attempt (2026-10-30) showed that each Triton
# specialization of M (M = 1, M % 16 == 0, other M) recompiles all 640 configs of the upstream
# search space at about 1.5 s each, so the full list cannot fit one container. The retry calls the
# script's own tune() unmodified, with the search space filtered to BLOCK_SIZE_M = 16 (128 configs).
TUNE_BATCHES = (1, 2, 4, 8, 16, 32, 64)
TUNE_BLOCK_M = 16
TUNE_BUDGET_S = 16 * 60  # stop tuning here; the session needs about 10 minutes
# Prefill (M >= 2048) keeps SGLang's built-in default config, stored under this key.
TUNE_DEFAULT_KEY = 4096
TUNE_DEFAULT = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32,
                "num_warps": 4, "num_stages": 3}  # fp8_kernel.py v0.5.22, no-config branch
TUNE_DRIVER = r"""
import importlib.util, json, sys, torch
spec = importlib.util.spec_from_file_location("tbk", sys.argv[1])
tbk = importlib.util.module_from_spec(spec); spec.loader.exec_module(tbk)
n, k, m, block_m = (int(x) for x in sys.argv[2:6])
space = [c for c in tbk.get_configs_compute_bound() if c["BLOCK_SIZE_M"] == block_m]
best = tbk.tune(m, n, k, [128, 128], torch.bfloat16, space, "fp8")
print("RESULT " + json.dumps({"M": m, "config": best, "search_space": len(space)}))
"""
TUNE_SCRIPT = "/sgl-workspace/sglang/benchmark/kernels/quantization/tuning_block_wise_kernel.py"
GEMM_CONFIG_DIR = "/sgl-workspace/sglang/python/sglang/kernels/ops/gemm/configs"
TRITON_DENSE_FLAGS = ["--thinker.engine.fp8_gemm_runner_backend", "triton",
                      "--thinker.engine.moe_runner_backend", "triton"]


@app.function(image=anatomy_image,
              volumes={WEIGHTS_MOUNT: weights, "/cache": jit_cache, "/results": results_vol},
              **GPU_LIMITS)
def fp8_tune(run_id: str, script_sha256: str) -> dict:
    """Tune Triton W8A8 block-FP8 configs with SGLang's own script, then one FP8 session using them."""
    import hashlib
    import os
    import re as _re
    import time
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    t0 = time.monotonic()
    out: dict = {"tuning": [], "configs": {}, "notes": [], "traces": {}}
    out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    dest = Path("/results/2c_fp8_tune") / run_id
    script = Path(TUNE_SCRIPT)
    out["script_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest() if script.is_file() else None
    out["script_matches_v0_5_22"] = out["script_sha256"] == script_sha256
    if not out["script_matches_v0_5_22"]:
        out["status"], out["error"] = "failed", f"tuning script missing or not v0.5.22: {out['script_sha256']}"
        return out
    import os as _os
    env = {**_os.environ, "TRITON_CACHE_DIR": "/cache/triton-tune"}  # kept for any retry
    tuned: dict = {shape: {} for shape in TUNE_SHAPES}
    for m in TUNE_BATCHES:
        for n, k in TUNE_SHAPES:
            remaining = TUNE_BUDGET_S - (time.monotonic() - t0)
            if remaining < 30:
                break
            t = time.monotonic()
            row = {"N": n, "K": k, "M": m}
            try:
                p = subprocess.run([SGLANG_PY, "-c", TUNE_DRIVER, TUNE_SCRIPT, str(n), str(k), str(m),
                                    str(TUNE_BLOCK_M)], capture_output=True, text=True, env=env,
                                   timeout=remaining)
                row["returncode"] = p.returncode
                res = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                if p.returncode == 0 and res:
                    r = json.loads(res[-1][7:])
                    tuned[(n, k)][m] = r["config"]
                    row["search_space"] = r["search_space"]
                else:
                    row["stderr"] = p.stderr[-3000:]
            except subprocess.TimeoutExpired:
                row["returncode"] = "timeout"
            row["seconds"] = round(time.monotonic() - t, 1)
            out["tuning"].append(row)
            print(f"[tune] N={n} K={k} M={m} rc={row['returncode']} {row['seconds']} s", flush=True)
        else:
            continue
        break
    try:
        jit_cache.commit()
    except Exception as exc:
        out["notes"].append(f"cache commit failed: {exc!r}")
    for (n, k), by_m in tuned.items():
        if by_m:
            name = f"N={n},K={k},device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json"
            out["configs"][name] = {**{str(m): c for m, c in sorted(by_m.items())},
                                    str(TUNE_DEFAULT_KEY): TUNE_DEFAULT}
    out["tuning_s"] = round(time.monotonic() - t0, 1)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "run.json").write_text(json.dumps(out, indent=2, default=str))
    results_vol.commit()
    if len(out["configs"]) != len(TUNE_SHAPES):
        out["status"], out["error"] = "failed", f"tuned files {sorted(out['configs'])}"
        return out
    for name, cfg in out["configs"].items():
        Path(GEMM_CONFIG_DIR, name).write_text(json.dumps(cfg, indent=4))

    raw = Path("/inputs/pg132_raw.txt").read_bytes()
    body_text = raw.decode("utf-8")
    words = body_text[body_text.index(TEXT_START_MARKER):].split()
    long_text = INSTRUCTION + " ".join(words[:6057])
    needles = build_needles(words)
    name = "B_triton_tuned"
    cmd = server_cmd(FP8_DIR, TRITON_DENSE_FLAGS)
    sess = {"server_cmd": cmd, "requests": []}
    out["session"] = sess
    trace_root = Path("/tmp/traces")

    def timed_request(tag: str) -> dict:
        rec = stream_chat(tag, long_text, MAX_TOKENS)
        rec["invalid"] = valid(rec)
        rec["text_sha256"] = hashlib.sha256(rec.pop("text").encode()).hexdigest()
        if rec["ttft_s"] is not None and rec["usage"]:
            rec["decode_ms_per_tok"] = ((rec["completion_s"] - rec["ttft_s"]) * 1000
                                        / max(1, rec["usage"]["completion_tokens"] - 1))
        rec.pop("chunk_offsets_us", None)
        return rec

    log_path = Path(f"/tmp/server-{name}.log")
    log_file = log_path.open("w", buffering=1)
    t_launch = time.monotonic()
    server = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                              env={**os.environ, **ANATOMY_ENV}, start_new_session=True)
    try:
        ready = False
        while time.monotonic() - t_launch < SERVER_READY_DEADLINE_S:
            if server.poll() is not None:
                sess["error"] = f"server exited during startup with code {server.returncode}"
                break
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                    ready = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        sess["server_ready_s"] = round(time.monotonic() - t_launch, 1) if ready else None
        print(f"[{name}] ready={ready} {sess['server_ready_s']} s", flush=True)
        if not ready:
            raise RuntimeError("server not healthy before deadline")
        # Needles first: their first 2k-token requests are where 2b-2 saw DeepGEMM compiles.
        q = []
        for item in needles:
            t = time.monotonic()
            g = generate(item["prompt"], 16)
            q.append({"id": item["id"], "context_words": item["context_words"],
                      "status": g["status"], "text": g["text"], "nonfinite": g["nonfinite_logprobs"],
                      "correct": item["answer"] in (g["text"] or ""),
                      "latency_s": round(time.monotonic() - t, 4)})
        sess["needles"] = q
        print(f"[{name}] needles correct {sum(x['correct'] for x in q)}/{len(q)} "
              f"max latency {max(x['latency_s'] for x in q):.3f} s", flush=True)
        warm = timed_request(f"{run_id}-{name}-warm")
        warm["measured"] = False
        sess["requests"].append(warm)
        for i in range(BACKEND_REPEATS):
            rec = timed_request(f"{run_id}-{name}-{i}")
            rec["measured"] = True
            sess["requests"].append(rec)
            print(f"[{name}/{i}] decode={rec.get('decode_ms_per_tok')} invalid={rec['invalid']}", flush=True)
        tdir = trace_root / name
        httpx.post(f"http://127.0.0.1:{PORT}/start_profile", timeout=60, json={
            "run_id": f"{run_id}-{name}", "enable_torch": True,
            "trace_path_template": str(tdir) + "/{stage}"})
        rec = timed_request(f"{run_id}-{name}-trace")
        httpx.post(f"http://127.0.0.1:{PORT}/stop_profile", timeout=60, json={})
        rec["trace_gz_ready"] = wait_for_trace(tdir)
        sess["trace_request"] = rec
        t = time.monotonic()
        with ThreadPoolExecutor(CONCURRENCY) as pool:
            recs = list(pool.map(lambda i: timed_request(f"{run_id}-{name}-c{i}"), range(CONCURRENCY)))
        wall = time.monotonic() - t
        toks = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in recs)
        sess["concurrent"] = {"n": CONCURRENCY, "wall_s": round(wall, 3), "output_tokens": toks,
                              "output_tok_per_s": round(toks / wall, 1), "requests": recs}
        print(f"[{name}] c{CONCURRENCY}: {toks / wall:.0f} tok/s", flush=True)
        sess["status"] = out["status"] = "completed"
    except Exception as exc:
        sess["status"] = out["status"] = "failed"
        sess["error"] = repr(exc)[:2000]
    finally:
        if server.poll() is None:
            os.killpg(server.pid, 15)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, 9)
        log_file.close()
        log = log_path.read_text(errors="replace")
        sess["server_log"] = log[-3_000_000:]
        sess["log_facts"] = [l[:400] for l in log.splitlines() if _re.search(
            r"KV Cache is allocated|Load weight end|backend policy|Config file not found|"
            r"Using default|Using configuration|W8A8|Traceback|Error", l)][:200]
        dest.mkdir(parents=True, exist_ok=True)
        for path in sorted(trace_root.rglob("*.gz")) if trace_root.exists() else []:
            rel = str(path.relative_to(trace_root))
            target = dest / "traces" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            out["traces"][rel] = path.stat().st_size
        (dest / "run.json").write_text(json.dumps(out, indent=2, default=str))
        results_vol.commit()
        jit_cache.commit()
    return out


# ---- Experiment 3: cold start, zero to first token ----------------------------
# A separate deployed App, because memory snapshots exist only for deployed Apps:
#   uv run modal deploy projects/qwen3-omni-on-modal/app.py::cold_app
# The client (3_cold_start/client.py) calls it from a clean scale-to-zero state.

cold_app = modal.App("qwen3-omni-cold-start")
COLD_PROMPT_WORDS = 807  # experiment 1's S prompt (1,150 tokens)
COLD_TOKENS = 32
COLD_FRACTION = "0.90"  # experiment 4a's recommended memory fraction
COLD_MODELS = {"fp8": (FP8_DIR, COMBINED_FLAGS), "bf16": (BF16_DIR, [])}
# budget.py row 3: 10 min startup + 1 min run + 1 min idle per cold container.
COLD_LIMITS = dict(gpu="H100!", cpu=8, memory=65536, max_containers=1, retries=0,
                   startup_timeout=600, timeout=60, scaledown_window=60)
COLD_VOLUMES = {WEIGHTS_MOUNT: weights, "/cache": jit_cache}


def cache_env(root: str) -> dict:
    """Every compiled-kernel cache the server uses, under one root."""
    return {"TRITON_CACHE_DIR": f"{root}/triton", "FLASHINFER_WORKSPACE_BASE": f"{root}/flashinfer",
            "TORCHINDUCTOR_CACHE_DIR": f"{root}/inductor", "SGLANG_CACHE_DIR": f"{root}/sglang"}


def dir_state(root: str) -> dict:
    """File count, bytes, and a hash of (path, size) for each cache directory."""
    import hashlib

    out = {}
    for key, path in cache_env(root).items():
        p = Path(path)
        files = sorted(f for f in p.rglob("*") if f.is_file()) if p.exists() else []
        h = hashlib.sha256("".join(f"{f.relative_to(p)}:{f.stat().st_size}\n" for f in files).encode())
        out[key] = {"files": len(files), "bytes": sum(f.stat().st_size for f in files),
                    "listing_sha256": h.hexdigest()[:16]}
    return out


class ColdServer:
    """Server lifecycle shared by the cold-start classes; clocks per 3_cold_start/README.md."""

    def start(self, model: str, cache_root: str) -> None:
        import os
        import time

        import httpx

        self.t = {"entry_mono": time.monotonic(), "entry_wall": time.time()}
        self.offset = self.t["entry_wall"] - self.t["entry_mono"]  # one offset, measured once
        self.model, self.cache_root = model, cache_root
        self.task_id = os.environ.get("MODAL_TASK_ID")
        self.cache_before = dir_state(cache_root)
        model_dir, flags = COLD_MODELS[model]
        self.cmd = ["sgl-omni", "serve", "--model-path", model_dir, "--text-only",
                    "--host", "127.0.0.1", "--port", str(PORT), "--model-name", SERVED_MODEL,
                    "--thinker.engine.disable_radix_cache", "true",
                    "--thinker.engine.mem_fraction_static", COLD_FRACTION, *flags]
        self.env = {**ANATOMY_ENV, **cache_env(cache_root)}
        self.log_path = Path("/tmp/server.log")
        self.log_file = self.log_path.open("w", buffering=1)
        self.t["launch_mono"] = time.monotonic()
        self.server = subprocess.Popen(self.cmd, stdout=self.log_file, stderr=subprocess.STDOUT,
                                       env={**os.environ, **self.env}, start_new_session=True)
        while time.monotonic() - self.t["launch_mono"] < SERVER_READY_DEADLINE_S:
            if self.server.poll() is not None:
                break
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                    self.t["ready_mono"] = time.monotonic()
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)

    def log_phases(self) -> dict:
        """Server-log phase timestamps (container wall clock) converted to container monotonic."""
        import datetime as _dt
        import re as _re

        log = self.log_path.read_text(errors="replace") if self.log_path.exists() else ""
        marks = {"load_begin": r"Load weight begin", "load_end": r"Load weight end",
                 "prefill_graph_begin": r"Capture target prefill CUDA graph begin",
                 "prefill_graph_end": r"Capture target prefill CUDA graph end",
                 "decode_graph_begin": r"Capture target decode CUDA graph begin",
                 "decode_graph_end": r"Capture target decode CUDA graph end",
                 "warmup_done": r"Server warmup of .* finished"}
        out = {"facts": [l[:300] for l in log.splitlines() if _re.search(
            r"Load weight end|CUDA graph end|warmup of|DeepGEMM|deep_gemm|Traceback|Error", l)][:60]}
        for key, pat in marks.items():
            for line in log.splitlines():
                m = _re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})", line)
                if m and _re.search(pat, line):
                    # Log timestamps are local wall time in the container (UTC on Modal).
                    wall = _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").replace(
                        tzinfo=_dt.timezone.utc).timestamp()
                    out[key] = round(wall - self.offset, 3)
                    break
        m = _re.search(r"Load weight end\. elapsed=([\d.]+) s.*?mem usage=([\d.]+) GB", log)
        if m:
            out["load_s"], out["load_gb"] = float(m.group(1)), float(m.group(2))
            out["load_gb_per_s"] = round(float(m.group(2)) / float(m.group(1)), 3)
        out["log_tail"] = log[-20000:]
        return out

    def first_token(self, tag: str):
        """Yield once at the first streamed token, then once with the full record."""
        import hashlib
        import time

        import httpx

        t_call = time.monotonic()
        raw = Path("/inputs/pg132_raw.txt").read_bytes()
        body_text = raw.decode("utf-8")
        words = body_text[body_text.index(TEXT_START_MARKER):].split()
        prompt = INSTRUCTION + " ".join(words[:COLD_PROMPT_WORDS])
        rec = {"tag": tag, "task_id": self.task_id, "model": self.model, "cmd": self.cmd,
               "cache_root": self.cache_root, "cache_before": self.cache_before,
               "server_alive": self.server.poll() is None, "text": "", "usage": None,
               "finish_reason": None, "error": None}
        body = {"model": SERVED_MODEL, "messages": [{"role": "user", "content": prompt}],
                "modalities": ["text"], "max_tokens": COLD_TOKENS, "temperature": 0.0, "stream": True}
        t_first = None
        try:
            with httpx.Client(timeout=30, trust_env=False) as client:
                with client.stream("POST", f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body) as resp:
                    for line in resp.iter_lines():
                        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                            continue
                        chunk = json.loads(line[5:])
                        rec["usage"] = chunk.get("usage") or rec["usage"]
                        for ch in chunk.get("choices", []):
                            piece = (ch.get("delta") or {}).get("content")
                            if piece:
                                if t_first is None:
                                    t_first = time.monotonic()
                                    yield {"event": "first_token", "container_mono": t_first}
                                rec["text"] += piece
                            rec["finish_reason"] = ch.get("finish_reason") or rec["finish_reason"]
        except Exception as exc:
            rec["error"] = repr(exc)[:2000]
        t = dict(self.t, call_mono=t_call, first_token_mono=t_first, done_mono=time.monotonic())
        rec["clock"] = t
        rec["phases"] = self.log_phases()
        rec["spans_s"] = {k: (round(t[b] - t[a], 3) if t.get(a) and t.get(b) else None) for k, (a, b) in {
            "entry_to_launch": ("entry_mono", "launch_mono"), "launch_to_ready": ("launch_mono", "ready_mono"),
            "ready_to_call": ("ready_mono", "call_mono"), "call_to_first_token": ("call_mono", "first_token_mono"),
            "entry_to_first_token": ("entry_mono", "first_token_mono")}.items()}
        rec["text_sha256"] = hashlib.sha256(rec["text"].encode()).hexdigest()
        rec["cache_after"] = dir_state(self.cache_root)
        if self.cache_root.startswith("/cache"):
            jit_cache.commit()  # the primed or tuned cache persists for the next container
        yield {"event": "record", "record": rec}


@cold_app.cls(image=anatomy_image, volumes=COLD_VOLUMES, **COLD_LIMITS)
class ColdStart(ColdServer):
    """F1/B1 baseline (fresh per-container caches) and F2 startup-tuned (caches on the Volume)."""

    model: str = modal.parameter()
    condition: str = modal.parameter()

    @modal.enter()
    def boot(self):
        root = "/tmp/fresh-cache" if self.condition == "baseline" else f"/cache/warm-{self.model}"
        self.start(self.model, root)

    @modal.method(is_generator=True)
    def request(self, tag: str):
        yield from self.first_token(tag)


@cold_app.cls(image=anatomy_image, volumes=COLD_VOLUMES, enable_memory_snapshot=True,
              experimental_options={"enable_gpu_snapshot": True}, **COLD_LIMITS)
class ColdSnapshot(ColdServer):
    """F3: GPU memory snapshot taken after the FP8 server is ready (one attempt)."""

    @modal.enter(snap=True)
    def boot(self):
        import time

        self.start("fp8", "/cache/warm-fp8")
        self.snap_wall = time.time()

    @modal.enter(snap=False)
    def restore(self):
        import os
        import time

        import httpx

        self.restore_wall = time.time()
        self.restore_task_id = os.environ.get("MODAL_TASK_ID")
        # After a restore, monotonic times from the snapshot phase belong to another boot.
        self.t["restore_mono"] = time.monotonic()
        try:
            self.restore_health = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=10).status_code
        except Exception as exc:
            self.restore_health = repr(exc)[:500]

    @modal.method(is_generator=True)
    def request(self, tag: str):
        for ev in self.first_token(tag):
            if ev["event"] == "record":
                ev["record"]["snapshot"] = {
                    "snap_phase_wall": self.snap_wall, "restore_wall": self.restore_wall,
                    "restored_from_snapshot": self.restore_wall - self.snap_wall > 30,
                    "task_id_at_snap": self.task_id, "task_id_now": self.restore_task_id,
                    "health_after_restore": self.restore_health}
                c = ev["record"]["clock"]
                # The container's own life starts at the restore; pre-snapshot spans are another boot.
                ev["record"]["spans_s"]["restore_to_first_token"] = (
                    round(c["first_token_mono"] - c["restore_mono"], 3) if c.get("first_token_mono") else None)
            yield ev


def fetch_traces(run_id: str, names: dict, local: Path, prefix: str = "1_anatomy") -> None:
    """Copy the run's trace files from the results Volume to the local results folder."""
    for rel in names:
        target = local / "traces" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            for chunk in results_vol.read_file(f"{prefix}/{run_id}/traces/{rel}"):
                fh.write(chunk)


# ---- Experiment 2a: FP8 smoke and gate (B, C, then the BF16 reference A) ----

MEM_FRACTION = "0.784"  # explicit for every condition: the BF16 default from experiment 1
FP8_CONDITIONS = (  # (name, model dir, extra flags); A last, as the reference
    ("B", FP8_DIR, []),
    ("C", FP8_DIR, ["--thinker.engine.kv_cache_dtype", "fp8_e4m3"]),
    ("A", BF16_DIR, []),
)
GATE_TOKENS = 100
SHORT_PROMPTS = (  # free generation, not copy tasks
    "Explain what a KV cache is in an LLM inference server, in three sentences.",
    "List four differences between prefill and decode in transformer inference.",
    "Write a short Python function that returns the n-th Fibonacci number iteratively.",
)
SHORT_TOKENS = 128
# Ceiling matches budget.py row 2a: 10 min startup + 30 min run + 2 min idle.
SMOKE_LIMITS = dict(GPU_LIMITS)


def server_cmd(model_dir: str, extra: list[str]) -> list[str]:
    return ["sgl-omni", "serve", "--model-path", model_dir, "--text-only",
            "--host", "127.0.0.1", "--port", str(PORT), "--model-name", SERVED_MODEL,
            "--thinker.engine.disable_radix_cache", "true",
            "--thinker.engine.mem_fraction_static", MEM_FRACTION, *extra]


def generate(text: str, max_new: int) -> dict:
    """Non-streaming /generate with logprobs: token ids and logprobs for the gate."""
    import math

    import httpx

    body = {"model": SERVED_MODEL, "messages": [{"role": "user", "content": text}],
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new},
            "return_logprob": True}
    rec = {"status": "failed", "error": None, "text": None, "token_ids": None,
           "logprobs": None, "finish": None, "nonfinite_logprobs": None}
    try:
        resp = httpx.post(f"http://127.0.0.1:{PORT}/generate", json=body, timeout=REQUEST_DEADLINE_S)
        rec["http_status"] = resp.status_code
        if resp.status_code != 200:
            rec["error"] = resp.text[:2000]
            return rec
        data = resp.json()
        meta = data.get("meta_info") or {}
        rec["text"], rec["finish"] = data.get("text"), meta.get("finish_reason")
        rec["completion_tokens"] = meta.get("completion_tokens")
        raw = meta.get("output_token_logprobs") or []
        rec["raw_logprob_sample"] = raw[:3]
        ids, lps = [], []
        for item in raw:  # [logprob, token_id, text?] or {"logprob":..., "token_id":...}
            if isinstance(item, dict):
                lp, tid = item.get("logprob"), item.get("token_id")
            else:
                lp, tid = item[0], item[1] if len(item) > 1 else None
            lps.append(lp)
            ids.append(tid)
        rec["token_ids"], rec["logprobs"] = ids, lps
        rec["nonfinite_logprobs"] = sum(1 for x in lps if x is None or not math.isfinite(float(x)))
        rec["status"] = "completed" if rec["text"] else "empty"
    except Exception as exc:
        rec["error"] = repr(exc)[:2000]
    return rec


GPU_PROBE = r"""
import json, torch
x = torch.tensor([516.0, -516.0, 448.0, 1.0], dtype=torch.bfloat16, device="cuda")
cast = x.to(torch.float8_e4m3fn).float().tolist()
n = 1 << 30  # 2 GiB of bf16 per buffer
a = torch.empty(n, dtype=torch.bfloat16, device="cuda"); b = torch.empty_like(a)
for _ in range(3): b.copy_(a)
torch.cuda.synchronize()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(20): b.copy_(a)
e.record(); torch.cuda.synchronize()
sec = s.elapsed_time(e) / 1000
print(json.dumps({"cast_in": [516.0, -516.0, 448.0, 1.0], "cast_out_cuda": cast,
                  "copy_bytes_per_iter": 2 * a.numel() * 2, "copy_s": sec,
                  "copy_gb_per_s": 20 * 2 * a.numel() * 2 / sec / 1e9}))
"""


@app.function(image=anatomy_image,
              volumes={WEIGHTS_MOUNT: weights, "/cache": jit_cache, "/results": results_vol},
              **SMOKE_LIMITS)
def fp8_smoke(run_id: str) -> dict:
    """Gate B and C against A on one H100: cast, non-finite logprobs, token match, KV dtype."""
    import hashlib
    import os
    import re as _re
    import time

    import httpx

    out: dict = {"conditions": {}, "notes": []}
    out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    probe = sh([SGLANG_PY, "-c", GPU_PROBE])
    try:
        out["gpu_probe"] = json.loads(probe["stdout"].strip().splitlines()[-1])
    except Exception:
        out["gpu_probe"] = {"error": probe["stderr"][-2000:]}
    print(f"[gpu probe] {out['gpu_probe']}", flush=True)

    raw = Path("/inputs/pg132_raw.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEXT_SHA256:
        out["status"], out["error"] = "failed", "text hash mismatch"
        return out
    body_text = raw.decode("utf-8")
    words = body_text[body_text.index(TEXT_START_MARKER):].split()
    long_text = INSTRUCTION + " ".join(words[:6057])  # experiment 1's L prompt (8,000 tokens)
    out["long_prompt_sha256"] = hashlib.sha256(long_text.encode()).hexdigest()

    for name, model_dir, extra in FP8_CONDITIONS:
        cmd = server_cmd(model_dir, extra)
        cond = {"server_cmd": cmd, "steps": {}}
        out["conditions"][name] = cond
        log_path = Path(f"/tmp/server-{name}.log")
        log_file = log_path.open("w", buffering=1)
        t_launch = time.monotonic()
        server = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                  env={**os.environ, **ANATOMY_ENV}, start_new_session=True)
        try:
            ready = False
            while time.monotonic() - t_launch < SERVER_READY_DEADLINE_S:
                if server.poll() is not None:
                    cond["error"] = f"server exited during startup with code {server.returncode}"
                    break
                try:
                    if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                        ready = True
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            cond["server_ready_s"] = round(time.monotonic() - t_launch, 1) if ready else None
            print(f"[{name}] ready={ready} {cond['server_ready_s']} s", flush=True)
            if not ready:
                cond["status"] = "failed"
                continue
            cond["steps"]["gate_long"] = generate(long_text, GATE_TOKENS)
            for i, prompt in enumerate(SHORT_PROMPTS):
                cond["steps"][f"short_{i}"] = generate(prompt, SHORT_TOKENS)
            full = stream_chat(f"{run_id}-{name}-full_L", long_text, MAX_TOKENS)
            full["invalid"] = valid(full)
            full["text_sha256"] = hashlib.sha256(full.pop("text").encode()).hexdigest()
            if full["ttft_s"] is not None and full["usage"]:
                full["decode_ms_per_tok"] = ((full["completion_s"] - full["ttft_s"]) * 1000
                                             / max(1, full["usage"]["completion_tokens"] - 1))
            cond["steps"]["full_L"] = full
            g = cond["steps"]["gate_long"]
            print(f"[{name}] gate {g['status']} nonfinite={g['nonfinite_logprobs']} "
                  f"full_L invalid={full['invalid']} decode={full.get('decode_ms_per_tok')}", flush=True)
            cond["gpu_after"] = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                                "--format=csv,noheader,nounits"],
                                               capture_output=True, text=True).stdout.strip()
            cond["status"] = "completed"
        except Exception as exc:
            cond["status"], cond["error"] = "failed", repr(exc)[:2000]
        finally:
            if server.poll() is None:
                os.killpg(server.pid, 15)
                try:
                    server.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, 9)
                    server.wait(timeout=30)
            log_file.close()
            log = log_path.read_text(errors="replace")
            cond["server_log"] = log[-3_000_000:]
            cond["log_facts"] = [l[:400] for l in log.splitlines() if _re.search(
                r"KV Cache is allocated|Load weight end|backend policy|quantiz|DeepGEMM|deep_gemm|"
                r"Config file not found|Traceback|Error|\bnan\b|NaN", l)][:200]
            for _ in range(60):  # wait for GPU memory to come back before the next server
                used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True).stdout.strip()
                if used.isdigit() and int(used) < 2000:
                    break
                time.sleep(2)

    # Gate: token-level comparison against A.
    ref = out["conditions"].get("A", {}).get("steps", {})
    for name in ("B", "C"):
        cond = out["conditions"].get(name, {})
        steps = cond.get("steps", {})
        gate = {"reasons": []}
        cast = (out.get("gpu_probe") or {}).get("cast_out_cuda")
        gate["cast_saturates"] = cast is not None and cast[:3] == [448.0, -448.0, 448.0]
        if not gate["cast_saturates"]:
            gate["reasons"].append(f"cuda cast {cast}")
        for key in ["gate_long"] + [f"short_{i}" for i in range(len(SHORT_PROMPTS))]:
            mine, theirs = steps.get(key, {}), ref.get(key, {})
            a_ids, b_ids = theirs.get("token_ids") or [], mine.get("token_ids") or []
            prefix = next((i for i, (x, y) in enumerate(zip(a_ids, b_ids)) if x != y), min(len(a_ids), len(b_ids)))
            same = sum(x == y for x, y in zip(a_ids, b_ids))
            gate[key] = {"status": mine.get("status"), "nonfinite": mine.get("nonfinite_logprobs"),
                         "len": len(b_ids), "ref_len": len(a_ids), "first_divergence": prefix,
                         "positional_match": round(same / max(1, len(a_ids)), 3)}
            if mine.get("status") != "completed":
                gate["reasons"].append(f"{key} status {mine.get('status')}")
            if mine.get("nonfinite_logprobs"):
                gate["reasons"].append(f"{key} has {mine['nonfinite_logprobs']} non-finite logprobs")
        if gate["gate_long"]["positional_match"] < 0.5:
            gate["reasons"].append(f"gate_long positional match {gate['gate_long']['positional_match']} < 0.5")
        full = steps.get("full_L", {})
        if full.get("invalid"):
            gate["reasons"].append(f"full_L invalid {full['invalid']}")
        if name == "C":
            kv = [l for l in cond.get("log_facts", []) if "KV Cache is allocated" in l]
            gate["kv_line"] = kv[0] if kv else None
            if not kv or "float8" not in kv[0]:
                gate["reasons"].append("no FP8 KV pool in the server log")
        gate["pass"] = not gate["reasons"]
        out.setdefault("gates", {})[name] = gate
        print(f"[gate {name}] pass={gate['pass']} reasons={gate['reasons']}", flush=True)
    out["status"] = "completed"
    dest = Path("/results/2a_fp8_smoke") / run_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "run.json").write_text(json.dumps(out, indent=2, default=str))
    results_vol.commit()
    jit_cache.commit()
    return out


@app.local_entrypoint()
def main(stage: str = "cpu-check"):
    if stage not in STAGES:
        raise SystemExit(f"--stage must be one of {STAGES}")
    run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}-{stage}"
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    if stage == "anatomy":
        result = anatomy.remote(run_id)
    elif stage == "fp8-smoke":
        result = fp8_smoke.remote(run_id)
    elif stage == "fp8-backends":
        result = fp8_backends.remote(run_id)
    elif stage == "fp8-combined":
        result = fp8_combined.remote(run_id)
    elif stage == "capacity-scan":
        result = capacity_scan.remote(run_id)
    elif stage == "capacity-sweep":
        result = capacity_sweep.remote(run_id)
    elif stage == "fp8-tune":
        import hashlib
        script = subprocess.run(["git", "-C", str(Path.home() / "projects/upstream/sglang"), "show",
                                 "v0.5.22:benchmark/kernels/quantization/tuning_block_wise_kernel.py"],
                                capture_output=True, check=True).stdout
        result = fp8_tune.remote(run_id, hashlib.sha256(script).hexdigest())
    else:
        result = {"cpu-check": cpu_check, "download-fp8": download_fp8}[stage].remote()
    record = {"schema_version": 1, "run_id": run_id, "stage": stage, "started_utc": started,
              "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "omni_sha": OMNI_SHA, "base_image": BASE_IMAGE,
              "spend": None, "spend_reason": "per-run spend not exposed; see the cost log",
              "result": result}
    out = {"anatomy": ANATOMY_RESULTS, "fp8-smoke": FP8_SMOKE_RESULTS,
           "fp8-backends": FP8_BACKEND_RESULTS,
           "fp8-combined": FP8_COMBINED_RESULTS,
           "capacity-scan": CAPACITY_RESULTS, "fp8-tune": TUNE_RESULTS,
           "capacity-sweep": SWEEP_RESULTS}.get(stage, RESULTS) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str))
    if stage in ("anatomy", "fp8-backends", "fp8-combined", "fp8-tune") and result.get("traces"):
        fetch_traces(run_id, result["traces"], out, {"anatomy": "1_anatomy",
                     "fp8-backends": "2b_fp8_backends", "fp8-combined": "2b2_fp8_combined",
                     "fp8-tune": "2c_fp8_tune"}[stage])
    print(f"wrote {out / 'run.json'} status={result.get('status')} error={result.get('error')}")
