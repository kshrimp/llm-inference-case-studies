"""Experiment 3 client: one cold request per run against the deployed cold-start App.

Before each run it waits until the App has no container (scale to zero), so every
measured request lands on a new container. The client span (call -> first token
event) is on this machine's monotonic clock; container spans come back in the
record on the container's clock and are never subtracted from client times.

Usage (after `uv run modal deploy projects/qwen3-omni-on-modal/app.py::cold_app`):
  uv run python projects/qwen3-omni-on-modal/3_cold_start/client.py --cls ColdStart \
      --model fp8 --condition baseline --runs 3
"""

import argparse
import datetime as dt
import json
import subprocess
import time
import uuid
from pathlib import Path

import modal

APP = "qwen3-omni-cold-start"
OUT = Path("results/qwen3-omni-on-modal/3_cold_start")


def containers() -> list[dict]:
    raw = subprocess.run(["uv", "run", "modal", "container", "list", "--json"],
                         capture_output=True, text=True, check=True).stdout
    return [c for c in json.loads(raw) if c.get("app_name") == APP]


def wait_for_zero(limit_s: int = 300) -> list[dict]:
    """Wait until no container of the App is alive; return what was seen at the start."""
    seen = containers()
    end = time.monotonic() + limit_s
    while containers():
        if time.monotonic() > end:
            raise SystemExit(f"containers still alive after {limit_s} s; not starting a cold run")
        time.sleep(5)
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", choices=("ColdStart", "ColdSnapshot"), required=True)
    ap.add_argument("--model", choices=("fp8", "bf16"), default="fp8")
    ap.add_argument("--condition", choices=("baseline", "tuned", "prime", "snapshot"), required=True)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    cls = modal.Cls.from_name(APP, args.cls)
    if args.cls == "ColdStart":
        # "prime" fills the Volume cache with the tuned condition's settings; it is not measured.
        obj = cls(model=args.model, condition="baseline" if args.condition == "baseline" else "tuned")
    else:
        obj = cls()
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(args.runs):
        before = wait_for_zero()
        run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}-{args.condition}-{args.model}"
        rec = {"schema_version": 1, "run_id": run_id, "app": APP, "cls": args.cls, "model": args.model,
               "condition": args.condition, "index": i, "containers_alive_before_wait": before,
               "client": {"call_wall": time.time()}}
        t0 = time.monotonic()
        try:
            for ev in obj.request.remote_gen(run_id):
                if ev["event"] == "first_token":
                    rec["client"]["first_token_s"] = round(time.monotonic() - t0, 3)
                    print(f"[{run_id}] first token after {rec['client']['first_token_s']} s (client clock)", flush=True)
                elif ev["event"] == "record":
                    rec["container"] = ev["record"]
        except Exception as exc:  # a failed cold start (startup timeout, failed restore) is a result
            rec["client"]["error"] = repr(exc)[:2000]
        rec["client"]["done_s"] = round(time.monotonic() - t0, 3)
        c = rec.get("container") or {}
        span = (c.get("spans_s") or {}).get("entry_to_first_token")
        if c.get("snapshot"):
            span = c["spans_s"].get("restore_to_first_token")
        # Upper bound on time outside the container (scheduling, image, boot): not a measured phase.
        first = rec["client"].get("first_token_s")
        rec["unobserved_upper_bound_s"] = round(first - span, 3) if first is not None and span is not None else None
        (OUT / f"{run_id}.json").write_text(json.dumps(rec, indent=2, default=str))
        print(f"[{run_id}] container entry->first token {span} s; unobserved <= {rec['unobserved_upper_bound_s']} s; "
              f"text={c.get('text', '')[:60]!r} error={rec['client'].get('error') or c.get('error')}", flush=True)


if __name__ == "__main__":
    main()
