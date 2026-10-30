"""Experiment 3 report: per-request speech timeline and the A′ gate (local, no GPU).

Reads results/omni-pipeline-probe/speech/<run_id>/{run.json, <session>/{requests.jsonl,
events.jsonl, server.log}}. For every request it prints client TTFT/TTFA, audio
duration, steady-state RTF and playback slack, then the event timeline before the
first audio as offsets from client dispatch (all on time_ns). For a speech-smoke run it
also writes the gate checks; a check is true, false, or null with a reason, and null
never counts as a pass. Writes report.json beside run.json.

Usage: uv run python projects/omni-pipeline-probe/speech_report.py <run_dir>
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SAMPLES_PER_FRAME = 1920  # one codec frame at 24 kHz (thinker_session.py)
CACHE_LINE = re.compile(r"encoder_cache stage=(\S+) action=(\S+) req=(\S+)")
CACHED_TOKEN = re.compile(r"#cached-token:\s*(\d+)")
# Timeline marks, in pipeline order; (stage, event) with stage "code2wav" for the vocoder (verified in A′; the emit call passes stage=None but the runtime fills it).
MARKS = (
    ("admission", ("coordinator", "request_admission")),
    ("preprocess_end", ("preprocessing", "preprocess_end")),
    ("encoder_end", ("audio_encoder", "encoder_end")),
    ("thinker_prefill_start", ("thinker", "scheduler_prefill_start")),
    ("thinker_prefill_end", ("thinker", "scheduler_prefill_end")),
    ("thinker_first_chunk", ("thinker", "stage_first_stream_chunk_sent")),
    ("thinker_complete", ("thinker", "stage_complete")),
    ("talker_queue_enter", ("talker_ar", "scheduler_queue_enter")),
    ("talker_prefill_start", ("talker_ar", "scheduler_prefill_start")),
    ("talker_prefill_end", ("talker_ar", "scheduler_prefill_end")),
    ("talker_first_chunk", ("talker_ar", "stage_first_stream_chunk_sent")),
    ("code2wav_first_ingest", ("code2wav", "code2wav_first_ingest")),
    ("code2wav_first_window_ready", ("code2wav", "code2wav_first_window_ready")),
    ("code2wav_first_audio", ("code2wav", "code2wav_first_audio")),
    ("talker_complete", ("talker_ar", "stage_complete")),
    ("terminal", ("coordinator", "terminal_response")),
)


def check(value, reason=None) -> dict:
    return {"pass": value, "reason": reason}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def playback(rec: dict) -> dict:
    """Steady-state RTF and simulated playback slack from the client's audio arrivals."""
    chunks = [c for c in rec.get("audio_chunks", []) if c.get("samples")]
    if not chunks or not rec.get("audio_sample_rate"):
        return {"rtf": None, "min_slack_ms": None, "stall_ms": None,
                "reason": "no decoded audio"}
    sr = rec["audio_sample_rate"]
    t0, played, slack, stall = chunks[0]["t_s"], 0.0, [], 0.0
    for c in chunks:
        due = t0 + played + stall  # playback pauses while a late chunk is awaited
        slack.append((due - c["t_s"]) * 1000)
        if c["t_s"] > due:
            stall += c["t_s"] - due
        played += c["samples"] / sr
    rest = played - chunks[0]["samples"] / sr
    rtf = (chunks[-1]["t_s"] - t0) / rest if rest > 0 else None
    return {"rtf": rtf, "min_slack_ms": min(slack[1:]) if len(slack) > 1 else None,
            "stall_ms": stall * 1000, "chunks": len(chunks)}


def request_row(rec: dict, ev: dict, log_cache: dict) -> dict:
    """ev maps (stage, event) -> sorted timestamps for this request."""
    d0 = rec["dispatch_wall_ns"]
    first = {k: (ev[key][0] - d0) / 1e6 if ev.get(key) else None for k, key in MARKS}
    decode_ends = ev.get(("code2wav", "code2wav_decode_end"), [])
    frames_events = None
    meta = ev.get("_meta", {}).get(("code2wav", "code2wav_decode_end"), [])
    if meta:
        frames_events = sum(m.get("audio_samples", 0) for m in meta) / SAMPLES_PER_FRAME
    starts = ev.get(("code2wav", "code2wav_decode_start"), [])
    decode_sum = (sum(e - s for s, e in zip(starts, decode_ends)) / 1e6
                  if starts and len(starts) == len(decode_ends) else None)
    tok = (rec.get("usage") or {}).get("completion_tokens")
    tf, tc = first["thinker_first_chunk"], first["thinker_complete"]
    pf, pc = first["talker_first_chunk"], first["talker_complete"]
    frames_client = rec["audio_samples"] / SAMPLES_PER_FRAME if rec.get("audio_samples") else None
    return {
        "step": rec["step"], "condition": rec["condition"], "measured": rec.get("measured"),
        "status": rec["status"], "finish": rec.get("finish_reason"), "completion_tokens": tok,
        "ttft_ms": rec["ttft_s"] * 1000 if rec.get("ttft_s") is not None else None,
        "ttfa_ms": rec["ttfa_s"] * 1000 if rec.get("ttfa_s") is not None else None,
        "completion_ms": rec["completion_s"] * 1000, "audio_s": rec.get("audio_duration_s"),
        "audio_sha256": rec.get("audio_sha256"), **playback(rec), "marks_ms": first,
        "thinker_ms_per_token": (tc - tf) / (tok - 1) if tf is not None and tc is not None and tok and tok > 1 else None,
        "frames_client": frames_client, "frames_events": frames_events,
        "talker_ms_per_frame": ((pc - pf) / (frames_events - 1)
                                if pf is not None and pc is not None and frames_events and frames_events > 1 else None),
        "code2wav_decode_sum_ms": decode_sum, "code2wav_windows": len(decode_ends),
        "encoder_cache": log_cache.get(rec["request_id"]),
        "event_keys": sorted(f"{k[0]}:{k[1]}" for k in ev if isinstance(k, tuple)),
    }


def session_rows(sdir: Path) -> tuple[list[dict], str]:
    recs = load_jsonl(sdir / "requests.jsonl")
    events = load_jsonl(sdir / "events.jsonl")
    log = (sdir / "server.log").read_text(errors="replace") if (sdir / "server.log").exists() else ""
    cache: dict = defaultdict(list)
    for stage, action, req in CACHE_LINE.findall(log):
        if stage == "audio_encoder":
            cache[req].append(action)
    by_req: dict = defaultdict(lambda: defaultdict(list))
    for e in events:
        if "request_id" not in e:
            continue
        key = (e.get("stage"), e.get("event_name"))
        by_req[e["request_id"]][key].append(e["timestamp_ns"])
        by_req[e["request_id"]].setdefault("_meta", defaultdict(list))[key].append(e.get("metadata") or {})
    for ev in by_req.values():
        for k, v in ev.items():
            if k != "_meta":
                v.sort()
    rows = [request_row(r, by_req.get(r["request_id"], {}), cache) for r in recs]
    return rows, log


def fmt(v, nd=1):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def print_session(name: str, rows: list[dict]) -> None:
    print(f"\n== session {name} ==")
    print("step | cond | tok | finish | TTFT | TTFA | completion | audio s | RTF | min slack | stall | "
          "thinker ms/tok | talker ms/frame | frames ev/client | cache")
    for r in rows:
        print(" | ".join([r["step"], r["condition"], fmt(r["completion_tokens"]), fmt(r["finish"]),
                          fmt(r["ttft_ms"]), fmt(r["ttfa_ms"]), fmt(r["completion_ms"]),
                          fmt(r["audio_s"], 2), fmt(r["rtf"], 3), fmt(r["min_slack_ms"]),
                          fmt(r["stall_ms"]), fmt(r["thinker_ms_per_token"], 2),
                          fmt(r["talker_ms_per_frame"], 2),
                          f"{fmt(r['frames_events'])}/{fmt(r['frames_client'])}",
                          ",".join(r["encoder_cache"] or []) or "—"]))
    print("\nTimeline before first audio (ms from client dispatch):")
    for r in rows:
        if r["ttfa_ms"] is None:
            continue
        marks = sorted(((k, v) for k, v in r["marks_ms"].items()
                        if v is not None and v <= r["ttfa_ms"] + 1), key=lambda kv: kv[1])
        print(f"  {r['step']}: " + ", ".join(f"{k}={v:.1f}" for k, v in marks)
              + f", client_first_audio={r['ttfa_ms']:.1f}")
    groups = defaultdict(list)
    for r in rows:
        if r["measured"] and r["status"] == "completed":
            groups[r["condition"]].append(r)
    for cond, rs in sorted(groups.items()):
        vals = {k: [r[k] for r in rs if r[k] is not None] for k in ("ttft_ms", "ttfa_ms", "rtf")}
        print(f"  median {cond} (n={len(rs)}): " + ", ".join(
            f"{k}={statistics.median(v):.3g} [{min(v):.3g}–{max(v):.3g}]" for k, v in vals.items() if v))


def smoke_gate(run: dict, rows: list[dict], log: str) -> dict:
    result = run.get("result") or {}
    base = next((s for s in result.get("sessions", []) if s["name"] == "base"), {})
    by = {r["step"]: r for r in rows}
    checks = {"gpu_is_h100": check("H100" in (result.get("gpu") or ""), result.get("gpu")),
              "inputs_match_manifest": check(all((result.get("input_check") or {"x": False}).values())),
              "server_ready": check(bool(base.get("server_ready")), f"{base.get('server_ready_s')} s")}
    for r in rows:
        checks[f"completed:{r['step']}"] = check(r["status"] == "completed", r["status"])
    speech = [r for r in rows if r["condition"] != "T32"]
    checks["speech_audio_24k"] = check(
        bool(speech) and all(r["audio_s"] and r["ttfa_ms"] is not None for r in speech),
        {r["step"]: r["audio_s"] for r in speech})
    checks["talker_cap_not_hit"] = check(
        all((r["frames_client"] or 0) < 600 - 10 for r in speech), {r["step"]: r["frames_client"] for r in speech})
    expect = {"s010_v01_first": ["miss"], "s010_v01_repeat": ["hit"], "s010_v02": ["miss"],
              "s030_v01": ["miss"], "s030_v02_full": ["miss"], "t030_v03": ["miss"],
              "overhead_profile_off": ["miss"], "overhead_profile_on": ["miss"]}
    seen = {s: [a for a in (by[s]["encoder_cache"] or []) if a in ("miss", "hit")] if s in by else None
            for s in expect}
    checks["encoder_cache_pattern"] = check(all(seen[s] == expect[s] for s in expect), seen)
    cached = [int(x) for x in CACHED_TOKEN.findall(log)]
    checks["thinker_radix_off"] = check(bool(cached) and all(c == 0 for c in cached) if cached else None,
                                        f"{len(cached)} prefill lines, max cached {max(cached) if cached else None}")
    profiled = [r for r in rows if r["step"] != "overhead_profile_off"]
    checks["events_correlate"] = check(all(r["marks_ms"]["admission"] is not None for r in profiled),
                                       [r["step"] for r in profiled if r["marks_ms"]["admission"] is None])
    order = {}
    for r in speech:
        if r["step"] == "overhead_profile_off":
            continue
        m = r["marks_ms"]
        order[r["step"]] = (None if m["talker_prefill_start"] is None or m["thinker_complete"] is None
                            else m["talker_prefill_start"] >= m["thinker_complete"])
    checks["talker_waits_for_thinker"] = check(
        all(v is True for v in order.values()) if all(v is not None for v in order.values()) else None,
        order)
    match = {}
    for r in speech:
        if r["frames_events"] is None or r["frames_client"] is None:
            match[r["step"]] = None
        else:
            match[r["step"]] = abs(r["frames_events"] - r["frames_client"]) < 1e-6
    checks["frames_match_samples"] = check(
        all(v is True for v in match.values() if v is not None) and any(v is not None for v in match.values())
        if match else None, match)
    t = by.get("t030_v03")
    checks["text_request_no_talker"] = check(
        None if t is None else not any(k.startswith("talker_ar:") for k in t["event_keys"]),
        t and [k for k in t["event_keys"] if k.startswith("talker_ar")])
    a, b = by.get("s010_v01_first"), by.get("s010_v01_repeat")
    checks["info:seed_reproduces_audio"] = check(
        None if not (a and b and a["audio_sha256"]) else a["audio_sha256"] == b["audio_sha256"],
        "informational, not a gate")
    required = [k for k in checks if not k.startswith("info:")]
    return {"checks": checks, "gate_for_B": {"pass": all(checks[k]["pass"] is True for k in required),
                                            "failed": [k for k in required if checks[k]["pass"] is not True]}}


def main(run_dir: Path) -> None:
    run = json.loads((run_dir / "run.json").read_text())
    report = {"run_id": run["run_id"], "stage": run["stage"], "sessions": {}}
    for sess in (run.get("result") or {}).get("sessions", []):
        sdir = run_dir / sess["name"]
        rows, log = session_rows(sdir)
        print_session(sess["name"], rows)
        print(f"  ready {sess.get('server_ready_s')} s, GPU after ready {sess.get('gpu_after_ready')}, "
              f"status {sess.get('status')} {sess.get('error') or ''}")
        report["sessions"][sess["name"]] = {"rows": rows, "server_ready_s": sess.get("server_ready_s"),
                                            "gpu_after_ready": sess.get("gpu_after_ready"),
                                            "status": sess.get("status")}
        if run["stage"] == "speech-smoke" and sess["name"] == "base":
            gate = smoke_gate(run, rows, log)
            report["gate"] = gate
            print("\nGate checks:")
            for k, v in gate["checks"].items():
                print(f"  {k}: {v['pass']}  {json.dumps(v['reason'], default=str)[:300]}")
            print(f"GATE FOR B′: {gate['gate_for_B']}")
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
