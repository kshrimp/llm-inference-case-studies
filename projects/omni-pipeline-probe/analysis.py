"""Shared reader for measurement runs (containers B/C). Local, no GPU.

Turns results/omni-pipeline-probe/measure/<run_id>/ into one row per request:
client times, raw usage, encoder cache actions, event intervals on the shared
time_ns clock, and a valid/invalid verdict with reasons. The experiment
table.py scripts print from these rows; nothing here edits raw artifacts.
"""

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from smoke_report import CACHE_LINE, CACHED_TOKEN, EVENT_ORDER, INTERVALS

MEASURE_ROOT = Path("results/omni-pipeline-probe/measure")
OUTPUT_TOKENS = 32
TEXT_TARGET, TEXT_TOLERANCE = 409, 5
EXPECTED_AUDIO_PROMPT_TOKENS = {10: 149, 30: 409, 90: 1189}
WAV_NAME = re.compile(r"aow_(\d{3})s_v(\d{2})\.wav")
COMMON_EVENTS = {
    ("coordinator", "request_admission"),
    ("preprocessing", "preprocess_start"),
    ("preprocessing", "preprocess_end"),
    ("thinker", "scheduler_queue_enter"),
    ("thinker", "scheduler_prefill_start"),
    ("thinker", "scheduler_prefill_end"),
    ("thinker", "stage_first_stream_chunk_sent"),
    ("thinker", "stage_complete"),
    ("coordinator", "terminal_response"),
}
AUDIO_EVENTS = {("audio_encoder", "encoder_start"), ("audio_encoder", "encoder_end")}


def run_dirs() -> list[Path]:
    return sorted(d for d in MEASURE_ROOT.glob("*-measure") if (d / "run.json").exists())


def load_rows(run_dir: Path) -> tuple[dict, list[dict]]:
    run = json.loads((run_dir / "run.json").read_text())
    steps = [json.loads(l) for l in (run_dir / "requests.jsonl").read_text().splitlines() if l.strip()]
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines() if l.strip()]
    log = (run_dir / "server.log").read_text(errors="replace")
    actions = defaultdict(list)
    for stage, action, req in CACHE_LINE.findall(log):
        if stage.startswith("audio"):
            actions[req].append(action)
    cached = [int(n) for n in CACHED_TOKEN.findall(log)]
    ev_by_req = defaultdict(list)
    for e in events:
        if "request_id" in e:
            ev_by_req[e["request_id"]].append(e)
    run_info = {"run_id": run["run_id"], "status": run["status"],
                "gpu": (run.get("result") or {}).get("gpu"),
                "server_ready_s": (run.get("result") or {}).get("server_ready_s"),
                "text_control": (run.get("result") or {}).get("text_control"),
                "all_cached_tokens_zero": bool(cached) and all(c == 0 for c in cached),
                "prefill_lines": len(cached)}
    rows = []
    for s in steps:
        usage = s.get("usage") or {}
        m = WAV_NAME.fullmatch(s.get("wav") or "")
        first = {}
        for e in sorted(ev_by_req.get(s["request_id"], []), key=lambda e: e["timestamp_ns"]):
            first.setdefault((e.get("stage"), e["event_name"]), e)
        intervals = {label: (round((first[b]["timestamp_ns"] - first[a]["timestamp_ns"]) / 1e6, 3)
                             if a in first and b in first else None)
                     for label, a, b in INTERVALS}
        admit = first.get(("coordinator", "request_admission"))
        intervals["client_dispatch_to_admission"] = (
            round((admit["timestamp_ns"] - s["dispatch_wall_ns"]) / 1e6, 3) if admit else None)
        names = [n for n in EVENT_ORDER if n in first]
        missing_events = sorted((COMMON_EVENTS | (AUDIO_EVENTS if m else set())) - set(first))
        # Steady-state decode: median gap between successive thinker stream chunks
        # (one token per chunk, verified in container A).
        chunk_ts = sorted(e["timestamp_ns"] for e in ev_by_req.get(s["request_id"], [])
                          if e.get("stage") == "thinker" and e["event_name"] == "stage_stream_chunk_sent")
        gaps = [(b - a) / 1e6 for a, b in zip(chunk_ts, chunk_ts[1:])]
        row = {
            "run_id": run["run_id"], "step": s["step"], "experiment": s.get("experiment"),
            "measured": s.get("measured", False), "kind": "audio" if m else "text",
            "duration_s": int(m.group(1)) if m else None, "variant": int(m.group(2)) if m else None,
            "status": s["status"], "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "finish_reason": s.get("finish_reason"),
            "ttft_ms": round(s["ttft_s"] * 1e3, 3) if s.get("ttft_s") is not None else None,
            "completion_ms": round(s["completion_s"] * 1e3, 3) if s.get("completion_s") is not None else None,
            "cache_actions": actions.get(s["request_id"], []),
            "missing_events": ["/".join(pair) for pair in missing_events],
            "thinker_chunk_count": len(chunk_ts),
            "event_order_ok": all(first[a]["timestamp_ns"] <= first[b]["timestamp_ns"]
                                  for a, b in zip(names, names[1:])) and bool(names),
            **{f"{k}_ms": v for k, v in intervals.items()},
        }
        row["thinker_chunk_gap_median_ms"] = round(statistics.median(gaps), 3) if gaps else None
        cont = row["thinker_continuation_ms"]
        row["decode_ms_per_token"] = (round(cont / (row["completion_tokens"] - 1), 3)
                                      if cont is not None and (row["completion_tokens"] or 0) > 1 else None)
        reasons = []
        if row["status"] != "completed":
            reasons.append(f"status {row['status']}")
        if row["completion_tokens"] != OUTPUT_TOKENS or row["finish_reason"] != "length":
            reasons.append(f"output {row['completion_tokens']} tokens, finish {row['finish_reason']}")
        if row["kind"] == "audio" and ("miss" not in row["cache_actions"] or "hit" in row["cache_actions"]):
            reasons.append(f"encoder cache {row['cache_actions'] or 'no trace line'}")
        if row["kind"] == "text" and (row["prompt_tokens"] is None
                                      or abs(row["prompt_tokens"] - TEXT_TARGET) > TEXT_TOLERANCE):
            reasons.append(f"text prompt {row['prompt_tokens']} outside {TEXT_TARGET}±{TEXT_TOLERANCE}")
        if row["kind"] == "audio" and row["prompt_tokens"] != EXPECTED_AUDIO_PROMPT_TOKENS.get(row["duration_s"]):
            reasons.append(f"audio prompt {row['prompt_tokens']} inconsistent with {row['duration_s']} s")
        if row["missing_events"]:
            reasons.append(f"missing stage events {row['missing_events']}")
        if row["thinker_chunk_count"] != row["completion_tokens"]:
            reasons.append(f"thinker chunks {row['thinker_chunk_count']} vs output tokens {row['completion_tokens']}")
        if not (run.get("result") or {}).get("input_check") or not all((run["result"]["input_check"]).values()):
            reasons.append("input hashes not verified")
        if not run_info["all_cached_tokens_zero"]:
            reasons.append("radix cache reuse in server log")
        if not row["event_order_ok"]:
            reasons.append("events missing or out of order")
        row["valid"] = not reasons
        row["invalid_reasons"] = reasons
        rows.append(row)
    return run_info, rows


def median(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 3) if values else None


def fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def print_table(rows: list[dict], columns: list[tuple[str, str]]) -> None:
    print("| " + " | ".join(h for h, _ in columns) + " |")
    print("|" + "|".join(" --- " for _ in columns) + "|")
    for r in rows:
        print("| " + " | ".join(fmt(r.get(k)) for _, k in columns) + " |")
