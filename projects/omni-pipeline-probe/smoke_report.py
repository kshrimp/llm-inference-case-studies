"""Check a container A smoke run and decide whether container B may run (local, no GPU).

Reads results/omni-pipeline-probe/smoke/<run_id>/{run.json, requests.jsonl,
events.jsonl, server.log}, prints a report, and writes checks.json beside them.
A check is true, false, or null with a reason; null never counts as a pass.

Usage: uv run python projects/omni-pipeline-probe/smoke_report.py <run_dir>
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Expected encoder cache action per smoke step (mirrors app.SMOKE_STEPS).
EXPECTED_CACHE = {
    "audio010_v01_first": "miss", "audio010_v01_repeat": "hit", "audio010_v02": "miss",
    "audio030_v01": "miss", "audio090_v01": "miss",
    "overhead_profile_off": "miss", "overhead_profile_on": "miss",
}
# Order on one request's timeline as (stage, event); time_ns clock in every process.
# Several events (stage_aggregate_ready, stage_first_stream_chunk_sent) are emitted by
# more than one stage, so the stage is part of the key.
EVENT_ORDER = (
    ("coordinator", "request_admission"), ("preprocessing", "preprocess_start"),
    ("preprocessing", "preprocess_end"), ("audio_encoder", "encoder_start"),
    ("audio_encoder", "encoder_end"), ("mm_aggregate", "stage_aggregate_ready"),
    ("thinker", "scheduler_queue_enter"), ("thinker", "scheduler_prefill_start"),
    ("thinker", "scheduler_prefill_end"), ("thinker", "scheduler_first_emit"),
    ("thinker", "stage_first_stream_chunk_sent"), ("decode", "stage_first_stream_chunk_sent"),
    ("thinker", "stage_complete"), ("coordinator", "terminal_response"),
)
INTERVALS = (
    ("preprocess", ("preprocessing", "preprocess_start"), ("preprocessing", "preprocess_end")),
    ("audio_encoder", ("audio_encoder", "encoder_start"), ("audio_encoder", "encoder_end")),
    ("admission_to_thinker_queue", ("coordinator", "request_admission"), ("thinker", "scheduler_queue_enter")),
    ("thinker_queue", ("thinker", "scheduler_queue_enter"), ("thinker", "scheduler_prefill_start")),
    ("thinker_first_forward", ("thinker", "scheduler_prefill_start"), ("thinker", "scheduler_prefill_end")),
    ("thinker_first_output", ("thinker", "scheduler_prefill_start"), ("thinker", "stage_first_stream_chunk_sent")),
    ("thinker_continuation", ("thinker", "stage_first_stream_chunk_sent"), ("thinker", "stage_complete")),
    ("admission_to_terminal", ("coordinator", "request_admission"), ("coordinator", "terminal_response")),
)
CACHE_LINE = re.compile(r"encoder_cache stage=(\S+) action=(\S+) req=(\S+)")
NEW_TOKEN = re.compile(r"#new-token:\s*(\d+)")
CACHED_TOKEN = re.compile(r"#cached-token:\s*(\d+)")


def audio_tokens(num_samples: int) -> int:
    """Source-derived audio token count (models/qwen3_asr/audio_lengths.py), hop 160."""
    frames = num_samples // 160
    leave = frames % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (frames // 100) * 13


def check(value, reason=None) -> dict:
    return {"pass": value, "reason": reason}


def main(run_dir: Path) -> None:
    run = json.loads((run_dir / "run.json").read_text())
    result = run.get("result") or {}
    steps = [json.loads(l) for l in (run_dir / "requests.jsonl").read_text().splitlines() if l.strip()] \
        if (run_dir / "requests.jsonl").exists() else []
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines() if l.strip()] \
        if (run_dir / "events.jsonl").exists() else []
    log = (run_dir / "server.log").read_text(errors="replace") if (run_dir / "server.log").exists() else ""
    by_step = {s["step"]: s for s in steps}
    checks: dict = {}

    checks["gpu_is_h100"] = check("H100" in (result.get("gpu") or ""), result.get("gpu"))
    inputs = result.get("input_check") or {}
    checks["inputs_match_manifest"] = check(bool(inputs) and all(inputs.values()), inputs)
    checks["server_ready"] = check(bool(result.get("server_ready")),
                                   f"ready after {result.get('server_ready_s')} s")
    for s in steps:
        checks[f"completed:{s['step']}"] = check(s["status"] == "completed", s.get("error") or s["status"])
    done = [s for s in steps if s["status"] == "completed"]
    checks["usage_in_stream"] = check(bool(done) and all(s.get("usage") for s in done),
                                      "usage chunk missing" if done else "no completed steps")

    # Encoder cache: every measured audio request must miss; the repeat must hit.
    actions = defaultdict(list)
    for stage, action, req in CACHE_LINE.findall(log):
        if stage.startswith("audio"):
            actions[req].append(action)
    cache_rows = {}
    for step, want in EXPECTED_CACHE.items():
        rec = by_step.get(step)
        if not rec or not rec.get("request_id"):
            cache_rows[step] = check(None, "step not run")
            continue
        got = actions.get(rec["request_id"], [])
        ok = (want in got and "hit" not in got) if want == "miss" else (want in got)
        cache_rows[step] = check(ok if got else None, got or "no encoder_cache line for this request id")
    checks["encoder_cache_trace"] = cache_rows
    checks["encoder_cache_ids_seen"] = sorted(actions)[:20]

    # The init log line ("Init Unified Radix Cache") appears whether or not the tree is
    # disabled (sglang_backend/cache.py::create_tree_cache passes disable= to it), so
    # judge by behaviour: an identical repeated prompt must report zero cached tokens.
    cached = [int(n) for n in CACHED_TOKEN.findall(log)]
    repeat = by_step.get("audio010_v01_repeat") or {}
    repeat_ran = repeat.get("status") == "completed"
    checks["radix_cache_off"] = check(
        (all(c == 0 for c in cached) if cached and repeat_ran else None),
        {"cached_token_values": cached[:30], "identical_repeat_completed": repeat_ran,
         "init_lines": [l.strip()[:200] for l in log.splitlines() if "Radix Cache" in l][:3]})
    checks["kv_pool_lines"] = [l.strip()[:300] for l in log.splitlines()
                               if "KV Cache is allocated" in l or "max_total_num_tokens" in l][:5]

    # Events: correlation by request id, order on the shared time_ns clock, intervals.
    ev_by_req = defaultdict(list)
    for e in events:
        if "request_id" in e:
            ev_by_req[e["request_id"]].append(e)
    event_rows = {}
    for s in steps:
        if not s.get("request_id") or not s.get("profiled"):
            continue
        evs = ev_by_req.get(s["request_id"], [])
        first = {}
        for e in sorted(evs, key=lambda e: e["timestamp_ns"]):
            first.setdefault((e.get("stage"), e["event_name"]), e)
        names = [n for n in EVENT_ORDER if n in first]
        order_ok = all(first[a]["timestamp_ns"] <= first[b]["timestamp_ns"]
                       for a, b in zip(names, names[1:]))
        intervals = {}
        for label, a, b in INTERVALS:
            intervals[label] = (round((first[b]["timestamp_ns"] - first[a]["timestamp_ns"]) / 1e6, 3)
                                if a in first and b in first else None)
        admit = first.get(("coordinator", "request_admission"))
        if s.get("first_content_wall_ns") and admit:
            intervals["admission_to_client_first_content"] = round(
                (s["first_content_wall_ns"] - admit["timestamp_ns"]) / 1e6, 3)
        if admit:
            intervals["client_dispatch_to_admission"] = round(
                (admit["timestamp_ns"] - s["dispatch_wall_ns"]) / 1e6, 3)
        event_rows[s["step"]] = {
            "event_count": len(evs), "names_present": ["/".join(n) for n in names],
            "missing": ["/".join(n) for n in EVENT_ORDER if n not in first],
            "order_ok": order_ok, "intervals_ms": intervals,
            "encoder_meta": first.get(("audio_encoder", "encoder_start"), {}).get("metadata"),
            "stages": sorted({e.get("stage") for e in evs if e.get("stage")}),
        }
    checks["events"] = event_rows
    other_ids = sorted(set(ev_by_req) - {s.get("request_id") for s in steps})
    checks["event_ids_not_matching_client"] = other_ids[:20]
    rid_ok = bool(event_rows) and all(r["event_count"] > 0 for r in event_rows.values())
    checks["event_request_id_correlation"] = check(
        rid_ok if event_rows else None,
        "every profiled request has events under its client request_id" if rid_ok
        else f"unmatched ids: {other_ids[:5]}")

    # Prompt accounting: duration difference vs source formula, then engine prefill counts.
    def ptoks(step):
        rec = by_step.get(step) or {}
        return (rec.get("usage") or {}).get("prompt_tokens")
    p10, p30, p90 = ptoks("audio010_v02"), ptoks("audio030_v01"), ptoks("audio090_v01")
    want_30_10 = audio_tokens(480000) - audio_tokens(160000)
    checks["prompt_tokens"] = {s["step"]: (s.get("usage") or {}).get("prompt_tokens") for s in steps}
    checks["prompt_delta_30s_minus_10s"] = check(
        (p30 - p10 == want_30_10) if p10 is not None and p30 is not None else None,
        f"observed {None if p10 is None or p30 is None else p30 - p10}, formula {want_30_10}")
    if p90 is not None and p10 is not None:
        checks["prompt_delta_90s_minus_10s"] = check(
            p90 - p10 == audio_tokens(1440000) - audio_tokens(160000),
            f"observed {p90 - p10}, formula {audio_tokens(1440000) - audio_tokens(160000)}")
    engine_counts = [int(n) for n in NEW_TOKEN.findall(log)]
    prompts = {v for v in checks["prompt_tokens"].values() if v}
    checks["thinker_prefill_counts_in_log"] = check(
        (prompts <= set(engine_counts)) if engine_counts and prompts else None,
        {"engine_new_token_values": engine_counts[:30], "api_prompt_tokens": sorted(prompts)})

    off, on = by_step.get("overhead_profile_off"), by_step.get("overhead_profile_on")
    checks["profile_overhead_single_pair"] = {
        "off": off and {"ttft_s": off["ttft_s"], "completion_s": off["completion_s"],
                        "completion_tokens": (off.get("usage") or {}).get("completion_tokens")},
        "on": on and {"ttft_s": on["ttft_s"], "completion_s": on["completion_s"],
                      "completion_tokens": (on.get("usage") or {}).get("completion_tokens")},
        "note": "one pair, different variants of one 10 s clip; descriptive only",
    }
    mem = []
    for _, sample in result.get("gpu_samples", []):
        try:
            mem.append(int(sample.split(",")[0]))
        except (ValueError, IndexError):
            pass
    checks["gpu_memory_mib"] = {"after_ready": result.get("gpu_after_ready"),
                                "after_requests": result.get("gpu_after_requests"),
                                "peak_sampled": max(mem) if mem else None,
                                "samples": len(mem)}
    checks["outputs"] = {s["step"]: {"text": (s.get("text") or "")[:300],
                                     "finish_reason": s.get("finish_reason"),
                                     "completion_tokens": (s.get("usage") or {}).get("completion_tokens"),
                                     "ttft_s": s.get("ttft_s"), "completion_s": s.get("completion_s")}
                         for s in steps}

    # Gate for container B: every required check must be exactly true.
    required = {
        "inputs_match_manifest": checks["inputs_match_manifest"]["pass"],
        "server_ready": checks["server_ready"]["pass"],
        "all_steps_completed": all(s["status"] == "completed" for s in steps) and len(steps) > 0,
        "audio_reaches_encoder": all((r["encoder_meta"] or {}).get("modality") == "audio"
                                     for k, r in event_rows.items() if k.startswith("audio")) or None,
        "encoder_cache_pattern": all(r["pass"] is True for r in cache_rows.values()),
        "radix_cache_off": checks["radix_cache_off"]["pass"],
        "event_request_id_correlation": checks["event_request_id_correlation"]["pass"],
        "event_order": all(r["order_ok"] and r["event_count"] > 0
                           for r in event_rows.values()) if event_rows else None,
    }
    checks["gate_for_container_B"] = {
        "required": required,
        "pass": all(v is True for v in required.values()),
        "failed_or_unknown": [k for k, v in required.items() if v is not True],
    }
    (run_dir / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({k: checks[k] for k in ("gate_for_container_B", "encoder_cache_trace",
                                             "prompt_delta_30s_minus_10s", "radix_cache_off",
                                             "gpu_memory_mib")}, indent=2))
    for step, row in event_rows.items():
        print(step, row["intervals_ms"], "missing:", row["missing"])
    print(f"Wrote {run_dir / 'checks.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
