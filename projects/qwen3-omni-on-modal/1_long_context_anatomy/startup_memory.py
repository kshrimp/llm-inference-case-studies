"""Startup phases and GPU memory from experiment 1 server logs and nvidia-smi samples.

Server-log clock only (one container clock); phases are differences of log
timestamps. Usage: python startup_memory.py [run_dir ...]
"""
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "results/qwen3-omni-on-modal/1_anatomy"
MARKS = [  # (label, regex on a log line); first match wins
    ("stage env configured", r"Configured stage process env defaults"),
    ("weight load begin", r"Load weight begin"),
    ("weight load end", r"Load weight end"),
    ("KV pool allocated", r"KV Cache is allocated"),
    ("prefill graph end", r"Capture target prefill CUDA graph end"),
    ("decode graph end", r"Capture target decode CUDA graph end"),
    ("pipeline ready", r"Process pipeline ready"),
    ("warmup done / ready", r"Server warmup .* finished"),
]


def ts(line: str) -> dt.datetime | None:
    m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})", line)
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f") if m else None


def report(run_dir: Path) -> None:
    res = json.loads((run_dir / "run.json").read_text())["result"]
    lines = res["server_log"].splitlines()
    print(f"\n## {run_dir.name}: server_ready_s={res.get('server_ready_s')}")
    prev = None
    for label, pat in MARKS:
        hit = next((l for l in lines if re.search(pat, l)), None)
        t = ts(hit) if hit else None
        delta = f"+{(t - prev).total_seconds():6.1f} s" if t and prev else "        "
        extra = re.search(r"(elapsed=[\d.]+ s|#tokens: \d+|avail mem=[\d.]+ GB|finished in [\d.]+ s)", hit or "")
        print(f"{label:22s} {t.time() if t else None}  {delta}  {extra.group(0) if extra else ''}")
        prev = t or prev
    mem = [int(s[1].split(",")[0]) for s in res["gpu_samples"] if s[1].split(",")[0].strip().isdigit()]
    total = int(res["gpu"].split(",")[1])
    print(f"GPU memory: after ready {res['gpu_after_ready']} MiB; peak sampled {max(mem)} of {total} MiB "
          f"({len(mem)} samples, 1 s); never used {(total - max(mem)) / 1024:.1f} GiB")


if __name__ == "__main__":
    dirs = [Path(a) for a in sys.argv[1:]] or sorted(p for p in ROOT.iterdir()
                                                    if (p / "run.json").exists())
    for d in dirs:
        report(d)
