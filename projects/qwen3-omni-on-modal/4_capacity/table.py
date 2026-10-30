"""Experiment 4a numbers from saved records. Usage: python table.py [run_dir ...]

Closed-loop bursts: n concurrent 8k/1k requests sent together, client clock.
$/1M output tokens is a price-based estimate (budget.py GPU-container rate) at the
burst throughput, an upper bound on what an open-loop SLO-qualified rate can reach.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "results/qwen3-omni-on-modal/4a_capacity_scan"
GPU_CONTAINER_PER_HOUR = 4.838  # budget.py, prices read 2026-10-25


def report(run_dir: Path) -> None:
    res = json.loads((run_dir / "run.json").read_text())["result"]
    total = int(res["gpu"].split(",")[1])
    print(f"\n## {run_dir.name}  GPU {res['gpu']}")
    for name, s in res["sessions"].items():
        if s.get("status") != "completed":
            print(f"{name}: {s.get('status')} {s.get('error')}")
            continue
        kv = re.search(r"#tokens: (\d+)", s["server_log"])
        retract = len(re.findall(r"Retract requests", s["server_log"]))
        print(f"\n### {name}: mem_fraction_static {s['mem_fraction_static']}, KV pool {kv.group(1) if kv else None} tokens, "
              f"peak {s['peak_used_mib']} / {total} MiB ({s['peak_used_mib'] / total:.1%}), gate {'pass' if s['gate_pass'] else 'FAIL'}, "
              f"retractions {retract}")
        print("| Concurrent | Output tok/s | $/1M output tok (est.) | Time between tokens p50 (median req) | p95 (worst req) | TTFT p50 | TTFT max | Invalid |")
        print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for k, l in s["levels"].items():
            reqs = l["requests"]
            tt = sorted(q["ttft_s"] for q in reqs)
            usd = GPU_CONTAINER_PER_HOUR / (l["output_tok_per_s"] * 3600) * 1e6
            print(f"| {k} | {l['output_tok_per_s']} | {usd:.2f} | {st.median(q['tbt_ms_p50'] for q in reqs):.2f} ms | "
                  f"{max(q['tbt_ms_p95'] for q in reqs):.2f} ms | {tt[len(tt) // 2]:.2f} s | {tt[-1]:.2f} s | {len(l['invalid'])} |")


if __name__ == "__main__":
    for d in [Path(a) for a in sys.argv[1:]] or sorted(p for p in ROOT.iterdir() if p.is_dir()):
        report(d)
