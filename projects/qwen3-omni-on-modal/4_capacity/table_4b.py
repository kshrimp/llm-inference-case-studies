"""Experiment 4b numbers from saved records. Usage: uv run python table_4b.py [run_dir ...]

Every verdict is recomputed here from the raw request records with the runner's own
rate_summary (the frozen criteria in README.md), and checked against the stored one.
Client clock only; engine counts come from SGLang batch log lines (container clock)
and are reported separately. $ per 1,000 requests is a price-based estimate.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from app import rate_summary  # noqa: E402

ROOT = HERE.parents[2] / "results/qwen3-omni-on-modal/4b_capacity_sweep"
GPU_CONTAINER_PER_HOUR = 4.838  # budget.py, prices re-read 2026-10-30


def engine_counts(lines: list[str]) -> dict:
    run = [int(m.group(1)) for l in lines if (m := re.search(r"#running-req: (\d+)", l))]
    queue = [int(m.group(1)) for l in lines if (m := re.search(r"#queue-req: (\d+)", l))]
    return {"running_max": max(run, default=None), "queue_max": max(queue, default=None)}


def report(run_dir: Path) -> None:
    res = json.loads((run_dir / "run.json").read_text())["result"]
    print(f"\n## {run_dir.name}  GPU {res['gpu']}  prompts sized ok: {res.get('prompts_ok')} "
          f"({min(p['prompt_tokens'] for p in res['prompts'])}–{max(p['prompt_tokens'] for p in res['prompts'])} tokens)")
    for name, s in res["sessions"].items():
        print(f"\n### {name} (mem_fraction_static {s.get('mem_fraction_static')}): {s.get('status')} {s.get('error') or ''}")
        if s.get("status") != "completed":
            continue
        print(f"prompt counts match the sized set: {s.get('prompts_match')}; "
              f"closed-loop max used for rates: {s['max_tok_s']} tok/s (4a: {s['max_tok_s_4a']})")
        for n, l in s.get("closed_loop", {}).items():
            print(f"- closed loop c{n}: {l['output_tok_per_s']} tok/s, TTFT max {l['ttft_max_s']:.2f} s, "
                  f"invalid {len(l['invalid'])}, peak {l['peak_used_mib']} MiB")
        eng = engine_counts(s.get("batch_log", []))
        print(f"engine (server log, whole session): running max {eng['running_max']}, queue max {eng['queue_max']}")
        print("| × max | Offered req/s | Completed req/s | Sent / done | TTFT p50 / p95 / p99 (s) | "
              "TBT p50 / p95 / p99 (ms) | Output tok/s | Max in flight | $/1k req (est.) | Verdict |")
        print("| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |")
        best = None
        for r in s["rates"]:
            if r.get("status") == "skipped":
                print(f"| {r['fraction']} | | | | | | | | | skipped: {r['error']} |")
                continue
            x = rate_summary(r["requests"], r["queue_bound_hit"], r["server_alive"])
            assert x["verdict"] == r["summary"]["verdict"], (name, r["fraction"])
            t, g = x["ttft_s"], x["tbt_ms"]
            usd = GPU_CONTAINER_PER_HOUR / (x["offered_rps"] * 3600) * 1000 if x["offered_rps"] else None
            verdict = x["verdict"] + (f" ({'; '.join(x['fail_reasons'])})" if x["fail_reasons"] else "")
            print(f"| {r['fraction']} | {x['offered_rps']:.3f} | {x['completed_rps']:.3f} | {x['sent']} / {x['completed']} | "
                  f"{t['p50']:.2f} / {t['p95']:.2f} / {t['p99']:.2f} | {g['p50']:.1f} / {g['p95']:.1f} / {g['p99']:.1f} | "
                  f"{x['output_tok_per_s']} | {r['max_in_flight']} | {usd:.2f} | {verdict} |")
            if x["verdict"] == "pass":
                best = (r["fraction"], x["offered_rps"], usd)
        print(f"**Capacity (highest passing rate): {best[1]:.3f} req/s at {best[0]}× (≈ ${best[2]:.2f} per 1,000 requests)**"
              if best else "**No passing rate.**")


if __name__ == "__main__":
    for d in [Path(a) for a in sys.argv[1:]] or sorted(p for p in ROOT.iterdir() if p.is_dir()):
        report(d)
