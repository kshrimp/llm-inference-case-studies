"""Experiment 3 numbers from saved cold-start records. Usage: uv run python table.py

Client span (local clock) and container phases (container clock) are printed side by
side and never subtracted from each other, except the labelled upper bound on time
outside the container. Validity: 32 tokens, and identical text within each model.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "results/qwen3-omni-on-modal/3_cold_start"
ORDER = ("baseline", "prime", "tuned", "snapshot")


def phase(p: dict, a: str, b: str):
    return round(p[b] - p[a], 1) if p.get(a) is not None and p.get(b) is not None else None


def main() -> None:
    recs = [json.loads(f.read_text()) for f in sorted(ROOT.glob("*.json")) if f.name != "validation.json"]
    texts = defaultdict(set)
    for r in recs:
        c = r.get("container") or {}
        if (c.get("usage") or {}).get("completion_tokens") == 32 and r["condition"] != "prime":
            texts[r["model"]].add(c.get("text_sha256"))
    print("Distinct 32-token outputs per model (valid runs must agree):", {k: len(v) for k, v in texts.items()})
    print("| Condition | Model | Run | Client: call → first token | Container: entry → first token | Unobserved ≤ | "
          "Launch → ready | Spawn → load | Load (GB/s) | Prefill graphs | Decode graphs | Cache files before | Valid |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in sorted(recs, key=lambda r: (r["model"], ORDER.index(r["condition"]), r["run_id"])):
        c = r.get("container") or {}
        p, s = c.get("phases") or {}, c.get("spans_s") or {}
        t = c.get("clock") or {}
        launch = t.get("launch_mono")
        spawn_to_load = round(p["load_begin"] - launch, 1) if p.get("load_begin") and launch else None
        own = s.get("restore_to_first_token") if c.get("snapshot") else s.get("entry_to_first_token")
        files = sum(v["files"] for v in (c.get("cache_before") or {}).values())
        ok = ((c.get("usage") or {}).get("completion_tokens") == 32 and not c.get("error")
              and not r["client"].get("error") and len(texts[r["model"]]) == 1)
        load = f"{p.get('load_s')} s ({p.get('load_gb_per_s')})" if p.get("load_s") else None
        print(f"| {r['condition']} | {r['model']} | {r['run_id'][:16]} | {r['client'].get('first_token_s')} s | {own} s | "
              f"{r.get('unobserved_upper_bound_s')} s | {s.get('launch_to_ready')} s | {spawn_to_load} s | {load} | "
              f"{phase(p, 'prefill_graph_begin', 'prefill_graph_end')} s | {phase(p, 'decode_graph_begin', 'decode_graph_end')} s | "
              f"{files} | {'yes' if ok else 'NO: ' + str(r['client'].get('error') or c.get('error'))[:80]} |")
        if c.get("snapshot"):
            print(f"|  | snapshot detail | {c['snapshot']} |")


if __name__ == "__main__":
    main()
