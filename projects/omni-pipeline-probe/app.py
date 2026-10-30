"""Bounded Modal runner for the SGLang-Omni / Qwen3-Omni pipeline probe.

Stages: `cpu-check` (CPU) builds the pinned image and checks the installed stack
and CLI; `download` (CPU) fills the weights Volume at the pinned revision and
verifies every file against Hugging Face metadata; `smoke` is container A, one
H100 running the text-only server for the compatibility checks in README.md
(compatibility, not benchmark performance); `measure` is containers B/C, the
frozen experiment 1 and 2 request sequence on the same server setup.
Experiment 3 (speech output, colocated one-GPU speech server): `speech-cpu`
resolves the speech config and lever flags on CPU; `speech-smoke` is A′,
`speech-measure` is B′ (baseline + one torch trace), `speech-ab --lever L*`
is C′/D′ (baseline, lever, baseline server sessions in one container).

No upstream source is copied here. sglang-omni (Apache-2.0) is installed from one
exact commit, following its v0.1.9 docker/Dockerfile dependency recipe.
"""

import datetime as dt
import json
import re
import subprocess
import uuid
from pathlib import Path

import modal

OMNI_REPO = "https://github.com/sgl-project/sglang-omni.git"
OMNI_SHA = "8200f49934d66de591c035cab1c626c0600905fa"  # v0.1.9
BASE_IMAGE = ("lmsysorg/sglang@sha256:"
              "5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e")
SGLANG_PY = "/opt/sglang/bin/python3"
MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
WEIGHTS_MOUNT = "/weights"
MODEL_DIR = f"{WEIGHTS_MOUNT}/Qwen3-Omni-30B-A3B-Instruct"
RESULTS = Path("results/omni-pipeline-probe/setup")
SMOKE_RESULTS = Path("results/omni-pipeline-probe/smoke")
LOCAL_WAV_DIR = Path("results/omni-pipeline-probe/inputs/wav")
LOCAL_PASSAGE = Path("results/omni-pipeline-probe/inputs/text/passage.txt")
MEASURE_RESULTS = Path("results/omni-pipeline-probe/measure")
SPEECH_RESULTS = Path("results/omni-pipeline-probe/speech")
LOCAL_MANIFEST = Path(__file__).with_name("inputs_manifest.json")
STAGES = ("cpu-check", "download", "smoke", "measure",
          "speech-cpu", "speech-smoke", "speech-measure", "speech-ab")

if not re.fullmatch(r"[0-9a-f]{40}", OMNI_SHA) or not re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION):
    raise ValueError("pins must be full 40-character SHAs")

# Mirrors the v0.1.9 Dockerfile: project dependencies with its protobuf override,
# the DAC packages, then flashinfer-python 0.7.1 without dependencies. Skipped:
# UCX build (inter-process RDMA transport; text-only runs every stage in one
# process) and the audar-tts / fun-cosyvoice3 / minicpm-o extras and qwen-tts
# (other models). The upstream entrypoint that pulls `main` at start is not used.
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

app = modal.App("omni-pipeline-probe")
weights = modal.Volume.from_name("omni-hf-cache", create_if_missing=True)
jit_cache = modal.Volume.from_name("omni-jit-cache", create_if_missing=True)

# Smoke image: the same pinned stack plus the WAVs and their manifest, mounted at
# container start (not baked into a layer), and JIT caches on a Volume.
smoke_image = (
    image.env({"FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
               "TRITON_CACHE_DIR": "/cache/triton", "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor"})
    .add_local_dir(str(LOCAL_WAV_DIR), "/inputs")
    .add_local_file(str(LOCAL_PASSAGE), "/passage.txt")
    .add_local_file(str(LOCAL_MANIFEST), "/inputs_manifest.json")
)

CPU_LIMITS = dict(cpu=2, memory=4096, max_containers=1, retries=0, timeout=900,
                  scaledown_window=60)


def sh(cmd: list[str], timeout: int = 300, keep: int = 6000) -> dict:
    import os
    # Wide terminal so typer help does not truncate long flag names.
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, "COLUMNS": "200"})
    return {"cmd": cmd, "returncode": proc.returncode,
            "stdout": proc.stdout[-keep:], "stderr": proc.stderr[-keep:]}


@app.function(image=image, volumes={WEIGHTS_MOUNT: weights}, **CPU_LIMITS)
def cpu_check() -> dict:
    """Installed versions, CLI flags, and the resolved text-only config (no GPU)."""
    versions = sh([SGLANG_PY, "-c", (
        "import importlib.metadata as m, json, sys\n"
        "out = {'python': sys.version.split()[0]}\n"
        "for p in ('torch', 'sglang', 'sglang-omni', 'transformers', 'flashinfer-python',"
        " 'flash-attn-4', 'protobuf', 'nixl-cu13'):\n"
        "    try: out[p] = m.version(p)\n"
        "    except m.PackageNotFoundError: out[p] = None\n"
        "import sglang_omni; out['sglang_omni_file'] = sglang_omni.__file__\n"
        "print(json.dumps(out))")])
    head = sh(["git", "-C", "/opt/sglang-omni", "rev-parse", "HEAD"])
    serve_help = sh(["sgl-omni", "serve", "--help"], keep=200_000)
    have_local = Path(MODEL_DIR, "config.json").is_file()
    model_ref = MODEL_DIR if have_local else f"{MODEL_ID}@{MODEL_REVISION}"
    resolve_cmd = ["sgl-omni", "config", "resolve", "--model-path", model_ref, "--text-only",
                   "--thinker.engine.disable_radix_cache", "true"]
    resolved_diff = sh(resolve_cmd + ["--show", "diff"])
    resolved_full = sh(resolve_cmd + ["--show", "config"])
    help_text = serve_help["stdout"]
    return {
        "versions": versions, "omni_head": head["stdout"].strip(),
        "omni_head_matches_pin": head["stdout"].strip() == OMNI_SHA,
        "serve_help_flags": {f: f in help_text for f in (
            "--text-only", "--allowed-local-media-path", "--skip-server-warmup",
            "--mem-fraction-static", "--model-name")},
        "serve_help": serve_help, "model_ref": model_ref,
        "resolve_diff": resolved_diff, "resolve_full": resolved_full,
    }


@app.function(image=image, volumes={WEIGHTS_MOUNT: weights}, cpu=4, memory=8192,
              max_containers=1, retries=0, timeout=7200, scaledown_window=60)
def download() -> dict:
    """Download the pinned revision and verify sizes and LFS SHA-256 against the Hub."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise RuntimeError(f"Hub resolved {info.sha}, expected {MODEL_REVISION}")
    expected = {s.rfilename: {"size": s.size, "sha256": s.lfs.sha256 if s.lfs else None}
                for s in info.siblings}
    started = dt.datetime.now(dt.timezone.utc)
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL_DIR, max_workers=8)
    weights.commit()

    def check(name: str) -> dict:
        path = Path(MODEL_DIR, name)
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
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "local_dir": MODEL_DIR,
        "started_utc": started.isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "file_count": len(files), "total_bytes": sum(f["size"] or 0 for f in files),
        "all_ok": all(f["ok"] for f in files),
        "lfs_hash_checked": sum(f["sha256"] is not None for f in files),
        "non_lfs_size_only": [f["file"] for f in files if f["expected_sha256"] is None],
        "files": files,
    }


# ---- GPU sessions: container A smoke and containers B/C measurement --------

PORT = 8000
SERVED_MODEL = "qwen3-omni"
SERVER_READY_DEADLINE_S = 780   # load + upstream warmup; leaves ~7 min of the 1,200 s timeout
REQUEST_DEADLINE_S = 180
AUDIO_INSTRUCTION = "Transcribe the speech in this audio exactly."
SMOKE_MAX_TOKENS = 128
SMOKE_TEXT_PROMPT = "Write one sentence about the ocean."
# Billable bounds, used for the reservation: startup_timeout + timeout + scaledown.
GPU_LIMITS = dict(gpu="H100!", cpu=8, memory=65536, max_containers=1, retries=0,
                  startup_timeout=600, timeout=1200, scaledown_window=60)
PRICE_PER_S = {"H100": 0.001097, "cpu_core": 0.0000131, "mem_gib": 0.00000222}  # read 2026-10-25
SERVER_CMD = [
    "sgl-omni", "serve", "--model-path", MODEL_DIR, "--text-only",
    "--host", "127.0.0.1", "--port", str(PORT), "--model-name", SERVED_MODEL,
    "--allowed-local-media-path", "/inputs",
    "--thinker.engine.disable_radix_cache", "true",
]
SERVER_ENV = {"SGLANG_OMNI_TRACE_ENCODER_CACHE": "1", "SGLANG_OMNI_WARMUP_TIMEOUT": "600",
              "PYTHONUNBUFFERED": "1"}
# Smoke: (step id, WAV or None for text, request profile on?, expected encoder cache action)
# Order matters: v01 twice proves a hit is visible, v02 proves a changed sample misses.
SMOKE_STEPS = (
    ("text_short", None, True, None),
    ("audio010_v01_first", "aow_010s_v01.wav", True, "miss"),
    ("audio010_v01_repeat", "aow_010s_v01.wav", True, "hit"),
    ("audio010_v02", "aow_010s_v02.wav", True, "miss"),
    ("audio030_v01", "aow_030s_v01.wav", True, "miss"),
    ("audio090_v01", "aow_090s_v01.wav", True, "miss"),  # only if every earlier step completed
    ("overhead_profile_off", "aow_010s_v03.wav", False, "miss"),
    ("overhead_profile_on", "aow_010s_v04.wav", True, "miss"),
)
# Measurement (frozen after container A, README "Frozen settings for B"). Every
# request is greedy, capped at 32 tokens, and valid only if it stops at the cap.
MEASURE_MAX_TOKENS = 32
TEXT_INSTRUCTION = "Repeat this text exactly:\n\n"
TEXT_TARGET_TOKENS = 409   # audio030's thinker prompt tokens in container A
TEXT_TOLERANCE = 5
# (step id, WAV or None for the text control, measured?, experiment)
MEASURE_STEPS = (
    ("warm_text", None, False, "warmup"),
    ("warm_audio010", "aow_010s_v07.wav", False, "warmup"),
    ("warm_audio030", "aow_030s_v07.wav", False, "warmup"),
    ("warm_audio090", "aow_090s_v07.wav", False, "warmup"),
    ("e1_text_control_a", None, True, "exp1"),
    ("e1_audio030_v01", "aow_030s_v01.wav", True, "exp1"),
    ("e1_audio030_v02", "aow_030s_v02.wav", True, "exp1"),
    ("e1_text_control_b", None, True, "exp1"),
    ("e2_audio010_v01", "aow_010s_v01.wav", True, "exp2"),
    ("e2_audio030_v03", "aow_030s_v03.wav", True, "exp2"),
    ("e2_audio090_v01", "aow_090s_v01.wav", True, "exp2"),
    ("e2_audio090_v02", "aow_090s_v02.wav", True, "exp2"),
    ("e2_audio030_v04", "aow_030s_v04.wav", True, "exp2"),
    ("e2_audio010_v02", "aow_010s_v02.wav", True, "exp2"),
)


def worst_case_usd() -> float:
    seconds = GPU_LIMITS["startup_timeout"] + GPU_LIMITS["timeout"] + GPU_LIMITS["scaledown_window"]
    per_s = (PRICE_PER_S["H100"] + GPU_LIMITS["cpu"] * PRICE_PER_S["cpu_core"]
             + GPU_LIMITS["memory"] / 1024 * PRICE_PER_S["mem_gib"])
    return round(seconds * per_s, 2)


def stream_chat(rid: str, wav: str | None, text: str, max_tokens: int) -> dict:
    """One streaming chat request; client clocks are monotonic plus a wall-clock anchor."""
    import time

    import httpx

    if wav is None:
        content = text
    else:
        content = [{"type": "audio_url", "audio_url": {"url": f"/inputs/{wav}"}},
                   {"type": "text", "text": text}]
    body = {"model": SERVED_MODEL, "messages": [{"role": "user", "content": content}],
            "modalities": ["text"], "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "request_id": rid}
    rec = {"request_id": rid, "wav": wav, "status": "failed", "http_status": None,
           "max_tokens": max_tokens, "dispatch_wall_ns": time.time_ns(),
           "first_content_wall_ns": None, "end_wall_ns": None, "ttft_s": None,
           "completion_s": None, "finish_reason": None, "usage": None,
           "text": "", "content_chunks": 0, "error": None}
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
                            if rec["first_content_wall_ns"] is None:
                                rec["ttft_s"] = (time.monotonic_ns() - t0) / 1e9
                                rec["first_content_wall_ns"] = time.time_ns()
                            rec["text"] += piece
                            rec["content_chunks"] += 1
                        if choice.get("finish_reason"):
                            rec["finish_reason"] = choice["finish_reason"]
        rec["status"] = "completed" if rec["text"] else "empty"
    except Exception as exc:  # timeout, connection reset, bad JSON: keep the record
        rec["error"] = repr(exc)[:2000]
    finally:
        rec["completion_s"] = (time.monotonic_ns() - t0) / 1e9
        rec["end_wall_ns"] = time.time_ns()
    return rec


def profile_call(route: str, out: dict, body: dict) -> None:
    import httpx

    resp = httpx.post(f"http://127.0.0.1:{PORT}/{route}", timeout=30, json=body)
    out["notes"].append(f"{route}: {resp.status_code} {resp.text[:300]}")


def log_step(out: dict, rec: dict) -> None:
    out["steps"].append(rec)
    print(f"[{rec['step']}] {rec['status']} ttft={rec['ttft_s']} total={rec['completion_s']:.2f}s "
          f"usage={rec['usage']} finish={rec['finish_reason']}", flush=True)


def run_smoke_steps(run_id: str, out: dict) -> None:
    profile_n, profiling = 0, False
    for step, wav, want_profile, _ in SMOKE_STEPS:
        if step == "audio090_v01" and not all(r["status"] == "completed" for r in out["steps"]):
            out["steps"].append({"step": step, "request_id": None, "status": "skipped",
                                 "error": "an earlier step did not complete"})
            continue
        if want_profile and not profiling:
            profile_n += 1
            prof_id = f"{run_id}-p{profile_n}"
            profile_call("start_request_profile", out,
                         {"run_id": prof_id, "event_dir": f"/tmp/events/{prof_id}"})
            profiling = True
        elif not want_profile and profiling:
            profile_call("stop_request_profile", out, {})
            profiling = False
        text = SMOKE_TEXT_PROMPT if wav is None else AUDIO_INSTRUCTION
        rec = stream_chat(f"{run_id}-{step}", wav, text, SMOKE_MAX_TOKENS)
        rec.update(step=step, profiled=profiling)
        log_step(out, rec)
    if profiling:
        profile_call("stop_request_profile", out, {})


def size_text_control(run_id: str, out: dict) -> str:
    """Choose the passage prefix whose server-reported prompt is closest to the target.

    Binary search on word count with 1-token probe requests (unmeasured). The
    server's own `usage.prompt_tokens` is the count, so no local tokenizer or
    chat-template assumption is needed.
    """
    words = Path("/passage.txt").read_text().split()
    probes: dict[int, int | None] = {}

    def prompt_tokens(n: int) -> int | None:
        if n not in probes:
            rec = stream_chat(f"{run_id}-size-{n}", None,
                              TEXT_INSTRUCTION + " ".join(words[:n]), 1)
            probes[n] = (rec.get("usage") or {}).get("prompt_tokens")
        return probes[n]

    lo, hi = 1, len(words)
    while lo < hi:  # smallest n with prompt_tokens(n) >= target
        mid = (lo + hi) // 2
        count = prompt_tokens(mid)
        if count is None:
            break
        if count < TEXT_TARGET_TOKENS:
            lo = mid + 1
        else:
            hi = mid
    candidates = [n for n in (lo - 1, lo) if n in probes or 1 <= n <= len(words)]
    best = min(candidates, key=lambda n: abs((prompt_tokens(n) or 10**9) - TEXT_TARGET_TOKENS))
    out["text_control"] = {"words": best, "prompt_tokens": probes.get(best),
                           "target": TEXT_TARGET_TOKENS, "tolerance": TEXT_TOLERANCE,
                           "probes": dict(sorted(probes.items()))}
    print(f"[text sizing] words={best} prompt_tokens={probes.get(best)}", flush=True)
    return TEXT_INSTRUCTION + " ".join(words[:best])


def run_measure_steps(run_id: str, out: dict) -> None:
    text_control = size_text_control(run_id, out)
    profile_call("start_request_profile", out,
                 {"run_id": f"{run_id}-p1", "event_dir": f"/tmp/events/{run_id}-p1"})
    for step, wav, measured, experiment in MEASURE_STEPS:
        text = text_control if wav is None else AUDIO_INSTRUCTION
        rec = stream_chat(f"{run_id}-{step}", wav, text, MEASURE_MAX_TOKENS)
        rec.update(step=step, profiled=True, measured=measured, experiment=experiment)
        log_step(out, rec)
    profile_call("stop_request_profile", out, {})


@app.function(image=smoke_image, volumes={WEIGHTS_MOUNT: weights, "/cache": jit_cache},
              **GPU_LIMITS)
def gpu_session(run_id: str, mode: str) -> dict:
    """Start the text-only server, run the smoke or measurement steps, always tear down."""
    import hashlib
    import os
    import threading
    import time

    import httpx
    import soundfile as sf

    out: dict = {"mode": mode, "server_cmd": SERVER_CMD, "server_env": SERVER_ENV,
                 "steps": [], "gpu_samples": [], "events": {}, "notes": []}

    def nvsmi(fields: str) -> str:
        return subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip()

    out["gpu"] = nvsmi("name,memory.total,driver_version")
    # Inputs must match the manifest by decoded samples, not just file bytes.
    manifest = json.loads(Path("/inputs_manifest.json").read_text())
    expected = {Path(c["file"]).name: c["decoded_int16_sha256"] for c in manifest["clips"]}
    wavs = {s[1] for s in (SMOKE_STEPS if mode == "smoke" else MEASURE_STEPS) if s[1]}
    out["input_check"] = {}
    for wav in sorted(wavs):
        pcm, sr = sf.read(f"/inputs/{wav}", dtype="int16")
        out["input_check"][wav] = (sr == 16000 and hashlib.sha256(
            pcm.astype("<i2").tobytes()).hexdigest() == expected[wav])
    if mode == "measure":
        passage = Path("/passage.txt").read_bytes()
        out["input_check"]["passage.txt"] = (
            hashlib.sha256(passage).hexdigest() == manifest["text_passage"]["sha256"])
    if not all(out["input_check"].values()):
        out["status"] = "failed"
        out["error"] = "input hash mismatch"
        return out

    stop_sampling = threading.Event()

    def sample_gpu():
        while not stop_sampling.is_set():
            out["gpu_samples"].append([time.time_ns(), nvsmi("memory.used,utilization.gpu")])
            stop_sampling.wait(1.0)

    log_path = Path("/tmp/server.log")
    log_file = log_path.open("w", buffering=1)
    t_launch = time.monotonic()
    server = subprocess.Popen(SERVER_CMD, stdout=log_file, stderr=subprocess.STDOUT,
                              env={**os.environ, **SERVER_ENV}, start_new_session=True)
    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    event_root = Path("/tmp/events")
    try:
        ready = False
        while time.monotonic() - t_launch < SERVER_READY_DEADLINE_S:
            if server.poll() is not None:
                out["notes"].append(f"server exited during startup with code {server.returncode}")
                break
            elapsed = time.monotonic() - t_launch
            if int(elapsed) % 60 < 2:  # progress in the local `modal run` output
                tail = log_path.read_text(errors="replace").strip().splitlines()[-1:] or [""]
                print(f"[startup {elapsed:.0f}s] {tail[0][:200]}", flush=True)
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                    ready = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        out["server_ready"] = ready
        out["server_ready_s"] = round(time.monotonic() - t_launch, 1) if ready else None
        if not ready:
            out["status"] = "failed"
            out["error"] = "server not healthy before deadline"
            return out
        out["gpu_after_ready"] = nvsmi("memory.used,memory.total")
        if mode == "smoke":
            run_smoke_steps(run_id, out)
        else:
            run_measure_steps(run_id, out)
        time.sleep(2)  # let recorders flush
        out["gpu_after_requests"] = nvsmi("memory.used,memory.total")
        out["status"] = "completed"
    except Exception as exc:
        out["status"] = "failed"
        out["error"] = repr(exc)[:2000]
    finally:
        stop_sampling.set()
        if server.poll() is None:
            os.killpg(server.pid, 15)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, 9)
        out["server_returncode"] = server.returncode
        log_file.close()
        log = log_path.read_text(errors="replace")
        out["server_log_bytes"] = len(log)
        out["server_log"] = log[-4_000_000:]
        for path in sorted(event_root.rglob("*.jsonl")) if event_root.exists() else []:
            out["events"][str(path.relative_to(event_root))] = path.read_text()
        try:
            jit_cache.commit()
        except Exception as exc:
            out["notes"].append(f"jit cache commit failed: {exc!r}")
    return out


# ---- Experiment 3: speech output on the colocated one-GPU speech server -----

SPEECH_CONFIG = "/opt/sglang-omni/examples/configs/qwen3_omni_colocated_h100_bf16.yaml"
SPEECH_SERVER_CMD = [
    "sgl-omni", "serve", "--model-path", MODEL_DIR, "--config", SPEECH_CONFIG, "--colocate",
    "--host", "127.0.0.1", "--port", str(PORT), "--model-name", SERVED_MODEL,
    "--allowed-local-media-path", "/inputs",
    "--thinker.engine.disable_radix_cache", "true",
]
# Config levers (3_speech_output/README.md "Levers"); each is appended to the base command.
LEVERS = {
    "base": [],
    "L1": ["--talker_ar.factory.enable_partial_start", "true"],
    "L2": ["--code2wav.factory.initial_codec_chunk_frames", "2",
           "--talker_ar.factory.codec_coalesce_early_frames", "12"],
    "L3": ["--mps", "auto"],
}
SPEECH_SEED = 1234
TALKER_MAX_FRAMES = 600            # 48 s of audio; a request that reaches it is invalid
SPEECH_READY_DEADLINE_S = 900      # colocated load + upstream speech warmup
SPEECH_REQUEST_DEADLINE_S = 240
# Function timeouts per stage; the reservation is startup_timeout + timeout + scaledown.
SPEECH_TIMEOUTS = {"speech-smoke": 1800, "speech-measure": 1500, "speech-ab": 2400}
# (condition, modalities, thinker max_tokens)
SPEECH_CONDITIONS = {"S32": (["text", "audio"], 32), "Sfull": (["text", "audio"], 128),
                     "T32": (["text"], 32)}
# Smoke A′: (step id, WAV or None for text, condition, request events on?, expected encoder cache)
SPEECH_SMOKE_STEPS = (
    ("text_short", None, "T32", True, None),
    ("s010_v01_first", "aow_010s_v01.wav", "S32", True, "miss"),
    ("s010_v01_repeat", "aow_010s_v01.wav", "S32", True, "hit"),
    ("s010_v02", "aow_010s_v02.wav", "S32", True, "miss"),
    ("s030_v01", "aow_030s_v01.wav", "S32", True, "miss"),
    ("s030_v02_full", "aow_030s_v02.wav", "Sfull", True, "miss"),
    ("t030_v03", "aow_030s_v03.wav", "T32", True, "miss"),
    ("overhead_profile_off", "aow_010s_v03.wav", "S32", False, "miss"),
    ("overhead_profile_on", "aow_010s_v04.wav", "S32", True, "miss"),
)
# Measurement session (B′, and each session of C′/D′): (step id, WAV, condition, measured?)
SPEECH_MEASURE_STEPS = (
    ("warm_S32", "aow_030s_v07.wav", "S32", False),
    ("warm_T32", "aow_030s_v07.wav", "T32", False),
    ("warm_Sfull", "aow_030s_v07.wav", "Sfull", False),
    ("S32_v01", "aow_030s_v01.wav", "S32", True),
    ("T32_v02", "aow_030s_v02.wav", "T32", True),
    ("Sfull_v03", "aow_030s_v03.wav", "Sfull", True),
    ("S32_v04", "aow_030s_v04.wav", "S32", True),
    ("Sfull_v05", "aow_030s_v05.wav", "Sfull", True),
    ("T32_v06", "aow_030s_v06.wav", "T32", True),
    ("S32_v00", "aow_030s_v00.wav", "S32", True),
)
# B′ only: one unmeasured S32 under the torch profiler (v07 again, so its encoder is a cache hit).
TRACE_STEP = ("trace_S32", "aow_030s_v07.wav", "S32", False)


def speech_cmd(lever: str) -> list[str]:
    return SPEECH_SERVER_CMD + LEVERS[lever]


@app.function(image=image, volumes={WEIGHTS_MOUNT: weights}, **CPU_LIMITS)
def speech_cpu_check() -> dict:
    """Step 0′: the pinned speech config, and how the base and each lever resolve (no GPU)."""
    out = {"config_file": sh(["cat", SPEECH_CONFIG]),
           "serve_help_has_mps": "--mps" in sh(["sgl-omni", "serve", "--help"],
                                                keep=200_000)["stdout"],
           "resolve": {}}
    for lever, flags in LEVERS.items():
        cmd = ["sgl-omni", "config", "resolve", "--model-path", MODEL_DIR, "--config", SPEECH_CONFIG,
               "--thinker.engine.disable_radix_cache", "true", *flags]
        out["resolve"][lever] = {"diff": sh(cmd + ["--show", "diff"]),
                                 "config": sh(cmd + ["--show", "config"], keep=200_000)}
    return out


def stream_speech(rid: str, wav: str | None, text: str, condition: str) -> dict:
    """One streaming chat request with optional speech output.

    Records client TTFT (first text) and TTFA (first audio delta with samples), and
    every audio delta's arrival time and decoded length. The decoded audio is kept as
    int16 PCM so the caller can save one WAV per request.
    """
    import base64
    import io
    import time

    import httpx
    import numpy as np
    import soundfile as sf

    modalities, max_tokens = SPEECH_CONDITIONS[condition]
    content = text if wav is None else [
        {"type": "audio_url", "audio_url": {"url": f"/inputs/{wav}"}},
        {"type": "text", "text": text}]
    body = {"model": SERVED_MODEL, "messages": [{"role": "user", "content": content}],
            "modalities": modalities, "max_tokens": max_tokens, "temperature": 0.0,
            "seed": SPEECH_SEED, "stream": True, "request_id": rid}
    if "audio" in modalities:
        body.update(audio={"format": "wav"}, talker_max_new_tokens=TALKER_MAX_FRAMES)
    rec = {"request_id": rid, "wav": wav, "condition": condition, "modalities": modalities,
           "status": "failed", "http_status": None, "max_tokens": max_tokens,
           "dispatch_wall_ns": time.time_ns(), "first_content_wall_ns": None,
           "first_audio_wall_ns": None, "end_wall_ns": None, "ttft_s": None, "ttfa_s": None,
           "completion_s": None, "finish_reason": None, "usage": None, "text": "",
           "content_chunks": 0, "audio_chunks": [], "audio_samples": 0,
           "audio_sample_rate": None, "audio_duration_s": None, "audio_sha256": None,
           "error": None}
    pcm_parts: list = []
    t0 = time.monotonic_ns()
    try:
        with httpx.Client(timeout=SPEECH_REQUEST_DEADLINE_S, trust_env=False) as client:
            with client.stream("POST", f"http://127.0.0.1:{PORT}/v1/chat/completions",
                               json=body) as resp:
                rec["http_status"] = resp.status_code
                if resp.status_code != 200:
                    rec["error"] = resp.read().decode(errors="replace")[:2000]
                    return rec
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        rec["usage"] = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            if rec["first_content_wall_ns"] is None:
                                rec["ttft_s"] = (time.monotonic_ns() - t0) / 1e9
                                rec["first_content_wall_ns"] = time.time_ns()
                            rec["text"] += piece
                            rec["content_chunks"] += 1
                        audio = (delta.get("audio") or {}).get("data")
                        if audio:
                            t_rel = (time.monotonic_ns() - t0) / 1e9
                            raw = base64.b64decode(audio)
                            item = {"t_s": t_rel, "bytes": len(raw), "riff": raw[:4] == b"RIFF",
                                    "samples": None, "sample_rate": None}
                            try:
                                pcm, sr = sf.read(io.BytesIO(raw), dtype="int16")
                                if pcm.ndim > 1:
                                    pcm = pcm[:, 0]
                                item.update(samples=int(pcm.shape[0]), sample_rate=int(sr))
                                pcm_parts.append(pcm)
                                rec["audio_sample_rate"] = rec["audio_sample_rate"] or int(sr)
                                if pcm.shape[0] and rec["ttfa_s"] is None:
                                    rec["ttfa_s"] = t_rel
                                    rec["first_audio_wall_ns"] = time.time_ns()
                            except Exception as exc:  # keep undecodable deltas visible
                                item["decode_error"] = repr(exc)[:300]
                            rec["audio_chunks"].append(item)
                        if choice.get("finish_reason"):
                            rec["finish_reason"] = choice["finish_reason"]
        rec["status"] = "completed" if rec["text"] else "empty"
    except Exception as exc:  # timeout, connection reset, bad JSON: keep the record
        rec["error"] = repr(exc)[:2000]
    finally:
        rec["completion_s"] = (time.monotonic_ns() - t0) / 1e9
        rec["end_wall_ns"] = time.time_ns()
        if pcm_parts:
            import hashlib
            pcm = np.concatenate(pcm_parts).astype("<i2")
            rec["audio_samples"] = int(pcm.shape[0])
            if rec["audio_sample_rate"]:
                rec["audio_duration_s"] = pcm.shape[0] / rec["audio_sample_rate"]
            rec["audio_sha256"] = hashlib.sha256(pcm.tobytes()).hexdigest()
            rec["_pcm"] = pcm.tobytes()
    return rec


def run_server_session(name: str, lever: str, steps: list[tuple], run_id: str,
                       ready_deadline_s: int, trace: tuple | None = None) -> dict:
    """Start one speech server, run the steps, always tear it down; returns raw artifacts.

    `steps` rows are (step id, WAV, condition, events on?, measured?, expected cache).
    """
    import os
    import threading
    import time

    import httpx

    sess = {"name": name, "lever": lever, "server_cmd": speech_cmd(lever), "steps": [],
            "gpu_samples": [], "events": {}, "traces": {}, "wavs": {}, "notes": []}

    def nvsmi(fields: str) -> str:
        return subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip()

    def call(route: str, body: dict) -> None:
        try:
            resp = httpx.post(f"http://127.0.0.1:{PORT}/{route}", timeout=60, json=body)
            sess["notes"].append(f"{route}: {resp.status_code} {resp.text[:300]}")
        except httpx.HTTPError as exc:
            sess["notes"].append(f"{route}: {exc!r}")

    stop_sampling = threading.Event()

    def sample_gpu():
        while not stop_sampling.is_set():
            sess["gpu_samples"].append([time.time_ns(), nvsmi("memory.used,utilization.gpu")])
            stop_sampling.wait(1.0)

    sess["gpu_before_start"] = nvsmi("memory.used,memory.total")
    event_root, trace_root = Path(f"/tmp/events/{name}"), Path(f"/tmp/traces/{name}")
    log_path = Path(f"/tmp/server-{name}.log")
    log_file = log_path.open("w", buffering=1)
    t_launch = time.monotonic()
    server = subprocess.Popen(speech_cmd(lever), stdout=log_file, stderr=subprocess.STDOUT,
                              env={**os.environ, **SERVER_ENV}, start_new_session=True)
    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    try:
        ready = False
        while time.monotonic() - t_launch < ready_deadline_s:
            if server.poll() is not None:
                sess["notes"].append(f"server exited during startup with code {server.returncode}")
                break
            elapsed = time.monotonic() - t_launch
            if int(elapsed) % 60 < 2:
                tail = log_path.read_text(errors="replace").strip().splitlines()[-1:] or [""]
                print(f"[{name} startup {elapsed:.0f}s] {tail[0][:200]}", flush=True)
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                    ready = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        sess["server_ready"] = ready
        sess["server_ready_s"] = round(time.monotonic() - t_launch, 1) if ready else None
        print(f"[{name}] ready={ready} after {time.monotonic() - t_launch:.0f}s", flush=True)
        if not ready:
            sess["status"] = "failed"
            sess["error"] = "server not healthy before deadline"
            return sess
        sess["gpu_after_ready"] = nvsmi("memory.used,memory.total")
        profiling, prof_n = False, 0
        for step, wav, condition, want_events, measured, expect in steps:
            if want_events and not profiling:
                prof_n += 1
                call("start_request_profile", {"run_id": f"{run_id}-{name}-p{prof_n}",
                                               "event_dir": str(event_root / f"p{prof_n}")})
                profiling = True
            elif not want_events and profiling:
                call("stop_request_profile", {})
                profiling = False
            text = SMOKE_TEXT_PROMPT if wav is None else AUDIO_INSTRUCTION
            rec = stream_speech(f"{run_id}-{name}-{step}", wav, text, condition)
            pcm = rec.pop("_pcm", None)
            if pcm:
                sess["wavs"][step] = pcm
            rec.update(step=step, session=name, lever=lever, profiled=profiling,
                       measured=measured, expected_cache=expect)
            sess["steps"].append(rec)
            print(f"[{name}/{step}] {rec['status']} ttft={rec['ttft_s']} ttfa={rec['ttfa_s']} "
                  f"audio={rec['audio_duration_s']} total={rec['completion_s']:.2f}s "
                  f"usage={rec['usage']} finish={rec['finish_reason']}", flush=True)
        if profiling:
            call("stop_request_profile", {})
        if trace is not None:
            step, wav, condition, measured = trace
            call("start_profile", {"run_id": f"{run_id}-{name}-trace", "enable_torch": True,
                                   "event_dir": str(event_root / "trace"),
                                   "trace_path_template": str(trace_root) + "/{stage}"})
            rec = stream_speech(f"{run_id}-{name}-{step}", wav, AUDIO_INSTRUCTION, condition)
            rec.pop("_pcm", None)
            rec.update(step=step, session=name, lever=lever, profiled=True, measured=measured,
                       expected_cache="hit", torch_trace=True)
            sess["steps"].append(rec)
            call("stop_profile", {})
            time.sleep(20)  # traces export and gzip in the stage processes
        time.sleep(2)
        sess["gpu_after_requests"] = nvsmi("memory.used,memory.total")
        sess["status"] = "completed"
    except Exception as exc:
        sess["status"] = "failed"
        sess["error"] = repr(exc)[:2000]
    finally:
        stop_sampling.set()
        if server.poll() is None:
            os.killpg(server.pid, 15)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, 9)
                server.wait(timeout=30)
        sess["server_returncode"] = server.returncode
        log_file.close()
        log = log_path.read_text(errors="replace")
        sess["server_log_bytes"] = len(log)
        sess["server_log"] = log[-4_000_000:]
        for path in sorted(event_root.rglob("*.jsonl")) if event_root.exists() else []:
            sess["events"][str(path.relative_to(event_root))] = path.read_text()
        for path in sorted(trace_root.rglob("*")) if trace_root.exists() else []:
            if path.is_file() and path.stat().st_size < 400_000_000:
                sess["traces"][str(path.relative_to(trace_root))] = path.read_bytes()
        # The next session needs the GPU memory back.
        for _ in range(60):
            used = nvsmi("memory.used").split(",")[0].strip()
            if used.isdigit() and int(used) < 2000:
                break
            time.sleep(2)
        sess["gpu_after_teardown"] = nvsmi("memory.used,memory.total")
    return sess


@app.function(image=smoke_image, volumes={WEIGHTS_MOUNT: weights, "/cache": jit_cache},
              **{**GPU_LIMITS, "timeout": max(SPEECH_TIMEOUTS.values())})
def speech_gpu(run_id: str, stage: str, lever: str) -> dict:
    """A′ smoke, B′ baseline + trace, or C′/D′ base-lever-base sessions in one container."""
    import hashlib
    import time

    import soundfile as sf

    t_start = time.monotonic()
    out: dict = {"stage": stage, "lever": lever, "sessions": [], "notes": []}
    out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True).stdout.strip()
    manifest = json.loads(Path("/inputs_manifest.json").read_text())
    expected = {Path(c["file"]).name: c["decoded_int16_sha256"] for c in manifest["clips"]}
    wavs = {s[1] for s in SPEECH_SMOKE_STEPS + SPEECH_MEASURE_STEPS + (TRACE_STEP,) if s[1]}
    out["input_check"] = {}
    for wav in sorted(wavs):
        pcm, sr = sf.read(f"/inputs/{wav}", dtype="int16")
        out["input_check"][wav] = (sr == 16000 and hashlib.sha256(
            pcm.astype("<i2").tobytes()).hexdigest() == expected[wav])
    if not all(out["input_check"].values()):
        out["status"], out["error"] = "failed", "input hash mismatch"
        return out
    measure_rows = [(s, w, c, True, m, "miss" if m else None) for s, w, c, m in SPEECH_MEASURE_STEPS]
    if stage == "speech-smoke":
        smoke_rows = [(s, w, c, e, False, x) for s, w, c, e, x in SPEECH_SMOKE_STEPS]
        out["sessions"].append(run_server_session("base", "base", smoke_rows, run_id,
                                                  SPEECH_READY_DEADLINE_S))
        # MPS probe: does `--mps auto` start inside a Modal container? Only with time to spare.
        left = SPEECH_TIMEOUTS[stage] - (time.monotonic() - t_start)
        if left > 600:
            probe = [("mps_s010_v05", "aow_010s_v05.wav", "S32", True, False, "miss")]
            out["sessions"].append(run_server_session("mps_probe", "L3", probe, run_id,
                                                      int(min(480, left - 120))))
        else:
            out["notes"].append(f"MPS probe skipped: {left:.0f}s left")
    elif stage == "speech-measure":
        out["sessions"].append(run_server_session("base", "base", measure_rows, run_id,
                                                  SPEECH_READY_DEADLINE_S, trace=TRACE_STEP))
    else:  # speech-ab: ABA
        for name, lev in (("base1", "base"), (lever, lever), ("base2", "base")):
            sess = run_server_session(name, lev, measure_rows, run_id, SPEECH_READY_DEADLINE_S)
            out["sessions"].append(sess)
            if sess["status"] != "completed":
                out["notes"].append(f"stopped after failed session {name}")
                break
    out["status"] = ("completed" if out["sessions"] and all(
        s["status"] == "completed" for s in out["sessions"] if s["name"] != "mps_probe")
        else "failed")
    try:
        jit_cache.commit()
    except Exception as exc:
        out["notes"].append(f"jit cache commit failed: {exc!r}")
    return out


def speech_worst_case_usd(stage: str) -> float:
    seconds = GPU_LIMITS["startup_timeout"] + SPEECH_TIMEOUTS[stage] + GPU_LIMITS["scaledown_window"]
    per_s = (PRICE_PER_S["H100"] + GPU_LIMITS["cpu"] * PRICE_PER_S["cpu_core"]
             + GPU_LIMITS["memory"] / 1024 * PRICE_PER_S["mem_gib"])
    return round(seconds * per_s, 2)


def save_speech_result(out_dir: Path, result: dict) -> None:
    """One folder per server session: server.log, events.jsonl, requests.jsonl, wav/, trace/."""
    import numpy as np
    import soundfile as sf

    for sess in result.get("sessions", []):
        d = out_dir / sess["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "server.log").write_text(sess.pop("server_log", ""))
        with (d / "events.jsonl").open("w") as handle:
            for name, text in sess.pop("events", {}).items():
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = {"source_file": name, **json.loads(line)}
                    except json.JSONDecodeError:
                        row = {"source_file": name, "unparsed": line}
                    handle.write(json.dumps(row) + "\n")
        with (d / "requests.jsonl").open("w") as handle:
            for rec in sess.get("steps", []):
                handle.write(json.dumps(rec) + "\n")
        rates = {r["step"]: r.get("audio_sample_rate") for r in sess.get("steps", [])}
        for step, pcm in sess.pop("wavs", {}).items():
            (d / "wav").mkdir(exist_ok=True)
            sf.write(d / "wav" / f"{step}.wav", np.frombuffer(pcm, dtype="<i2"),
                     rates.get(step) or 24000, subtype="PCM_16")
        for rel, data in sess.pop("traces", {}).items():
            path = d / "trace" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        sess["gpu_samples_count"] = len(sess.get("gpu_samples", []))
        (d / "gpu_samples.json").write_text(json.dumps(sess.pop("gpu_samples", [])))


def git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {"repo_head": head.stdout.strip() or None, "repo_dirty": bool(dirty.stdout.strip())}


@app.local_entrypoint()
def main(stage: str = "cpu-check", dry_run: bool = False, lever: str = ""):
    if stage not in STAGES:
        raise ValueError(f"--stage must be one of {STAGES}")
    if stage.startswith("speech-"):
        return speech_main(stage, dry_run, lever)
    gpu = stage in ("smoke", "measure")
    steps = SMOKE_STEPS if stage == "smoke" else MEASURE_STEPS
    run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}-{stage}"
    if dry_run:
        print(json.dumps({"gpu_launched": False, "stage": stage, "run_id": run_id,
                          "limits": GPU_LIMITS if gpu else None,
                          "estimated_max_usd": worst_case_usd() if gpu else None,
                          "server_cmd": SERVER_CMD if gpu else None,
                          "steps": [s[:2] for s in steps] if gpu else None}, indent=2))
        return
    record = {"schema_version": 1, "run_id": run_id, "stage": stage, "place": "modal",
              "gpu": GPU_LIMITS["gpu"] if gpu else None, "omni_sha": OMNI_SHA,
              "base_image": BASE_IMAGE, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **git_state()}
    if gpu:
        record.update(
            limits=GPU_LIMITS, estimated_max_usd=worst_case_usd(),
            measurement=("smoke: compatibility only, not benchmark performance" if stage == "smoke"
                         else "client-observed localhost single requests + in-engine request events"),
            max_tokens=SMOKE_MAX_TOKENS if stage == "smoke" else MEASURE_MAX_TOKENS,
            audio_instruction=AUDIO_INSTRUCTION,
            text_prompt=SMOKE_TEXT_PROMPT if stage == "smoke" else TEXT_INSTRUCTION + "<passage prefix>",
            steps=[list(s) for s in steps],
            inputs_manifest_sha256=__import__("hashlib").sha256(LOCAL_MANIFEST.read_bytes()).hexdigest())
    try:
        if stage == "cpu-check":
            record["result"] = cpu_check.remote()
        elif stage == "download":
            record["result"] = download.remote()
        else:
            record["result"] = gpu_session.remote(run_id, stage)
        record["status"] = record["result"].get("status", "completed")
    except Exception as exc:  # keep a record of failed attempts
        record["status"], record["error"] = "failed", repr(exc)
    record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    base = {"smoke": SMOKE_RESULTS, "measure": MEASURE_RESULTS}.get(stage, RESULTS)
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    result = record.get("result") or {}
    if gpu and result:
        # Raw artifacts as separate files; run.json keeps the structured record.
        (out / "server.log").write_text(result.pop("server_log", ""))
        events = result.pop("events", {})
        with (out / "events.jsonl").open("w") as handle:
            for name, text in events.items():
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = {"source_file": name, **json.loads(line)}
                    except json.JSONDecodeError:
                        row = {"source_file": name, "unparsed": line}
                    handle.write(json.dumps(row) + "\n")
        with (out / "requests.jsonl").open("w") as handle:
            for rec in result.get("steps", []):
                handle.write(json.dumps(rec) + "\n")
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {out / 'run.json'} status={record['status']}")
    if stage == "smoke":
        print(f"Next: uv run python projects/omni-pipeline-probe/smoke_report.py {out}")
    if record["status"] != "completed":
        raise SystemExit(1)


def speech_main(stage: str, dry_run: bool, lever: str) -> None:
    """Experiment 3 entry: speech-cpu, speech-smoke (A′), speech-measure (B′), speech-ab (C′/D′)."""
    if stage == "speech-ab" and lever not in LEVERS.keys() - {"base"}:
        raise ValueError(f"speech-ab needs --lever one of {sorted(LEVERS.keys() - {'base'})}")
    gpu = stage != "speech-cpu"
    suffix = f"-{lever}" if stage == "speech-ab" else ""
    run_id = (f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
              f"-{stage}{suffix}")
    sessions = {"speech-smoke": ["base", "mps_probe (L3, if time)"], "speech-measure": ["base"],
                "speech-ab": ["base", lever, "base"]}.get(stage)
    limits = {**GPU_LIMITS, "timeout": SPEECH_TIMEOUTS[stage]} if gpu else None
    if dry_run:
        print(json.dumps({"gpu_launched": False, "stage": stage, "run_id": run_id,
                          "limits": limits,
                          "estimated_max_usd": speech_worst_case_usd(stage) if gpu else None,
                          "sessions": sessions,
                          "server_cmds": {lev: speech_cmd(lev) for lev in LEVERS},
                          "steps": ([s[:3] for s in SPEECH_SMOKE_STEPS] if stage == "speech-smoke"
                                    else [s[:3] for s in SPEECH_MEASURE_STEPS] if gpu else None)},
                         indent=2))
        return
    record = {"schema_version": 1, "run_id": run_id, "stage": stage, "place": "modal",
              "gpu": GPU_LIMITS["gpu"] if gpu else None, "omni_sha": OMNI_SHA,
              "base_image": BASE_IMAGE, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "speech_config": SPEECH_CONFIG,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **git_state()}
    if gpu:
        record.update(
            limits=limits, estimated_max_usd=speech_worst_case_usd(stage), sessions=sessions,
            levers=LEVERS, lever=lever or None,
            measurement=("speech smoke: compatibility only, not benchmark performance"
                         if stage == "speech-smoke" else
                         "client-observed localhost single requests + in-engine request events"),
            conditions=SPEECH_CONDITIONS, seed=SPEECH_SEED, talker_max_new_tokens=TALKER_MAX_FRAMES,
            audio_instruction=AUDIO_INSTRUCTION, text_prompt=SMOKE_TEXT_PROMPT,
            steps=[list(s) for s in (SPEECH_SMOKE_STEPS if stage == "speech-smoke"
                                     else SPEECH_MEASURE_STEPS)],
            trace_step=list(TRACE_STEP) if stage == "speech-measure" else None,
            inputs_manifest_sha256=__import__("hashlib").sha256(LOCAL_MANIFEST.read_bytes()).hexdigest())
    try:
        if stage == "speech-cpu":
            record["result"] = speech_cpu_check.remote()
        else:
            fn = speech_gpu.with_options(timeout=SPEECH_TIMEOUTS[stage])
            record["result"] = fn.remote(run_id, stage, lever)
        record["status"] = record["result"].get("status", "completed")
    except Exception as exc:  # keep a record of failed attempts
        record["status"], record["error"] = "failed", repr(exc)
    record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    out = (RESULTS if stage == "speech-cpu" else SPEECH_RESULTS) / run_id
    out.mkdir(parents=True, exist_ok=True)
    if gpu and record.get("result"):
        save_speech_result(out, record["result"])
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {out / 'run.json'} status={record['status']}")
    if record["status"] != "completed":
        raise SystemExit(1)
