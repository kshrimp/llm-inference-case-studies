"""Experiment 2c numbers from saved records. Usage: uv run python table_2c.py [run_dir ...]

Tuning rows, the tuned tiles, batch-1 decode, needle latencies by context, and the
8-concurrent burst. The trace anatomy comes from ../1_long_context_anatomy/analyze_trace.py.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "results/qwen3-omni-on-modal/2c_fp8_tune"


def report(run_dir: Path) -> None:
    res = json.loads((run_dir / "run.json").read_text())["result"]
    print(f"\n## {run_dir.name}: {res.get('status')} {res.get('error') or ''}")
    for t in res["tuning"]:
        print(f"- tune N={t['N']} K={t['K']} M={t['M']}: rc={t['returncode']} {t['seconds']} s")
    for name, cfg in res.get("configs", {}).items():
        print(f"- {name}: " + ", ".join(f"M={m}: {c['BLOCK_SIZE_M']}x{c['BLOCK_SIZE_N']}x{c['BLOCK_SIZE_K']} "
                                         f"g{c['GROUP_SIZE_M']} w{c['num_warps']} s{c['num_stages']}" for m, c in cfg.items()))
    s = res.get("session")
    if not s:
        return
    m = [r for r in s["requests"] if r.get("measured")]
    print(f"\nDecode ms/token (8k, batch 1): median {st.median(r['decode_ms_per_tok'] for r in m):.3f} "
          f"[{min(r['decode_ms_per_tok'] for r in m):.3f}–{max(r['decode_ms_per_tok'] for r in m):.3f}], "
          f"TTFT median {st.median(r['ttft_s'] for r in m) * 1000:.1f} ms, invalid {sum(bool(r['invalid']) for r in m)}")
    for words in (1500, 3000, 6000):
        q = [x for x in s["needles"] if x["context_words"] == words]
        lat = [x["latency_s"] for x in q]
        print(f"Needles {words} words: {sum(x['correct'] for x in q)}/{len(q)} correct; latency median {st.median(lat):.3f} s, "
              f"max {max(lat):.3f} s; first six {[round(x, 3) for x in lat[:6]]}")
    c = s["concurrent"]
    print(f"{c['n']} concurrent 8k/1k: {c['output_tok_per_s']} output tok/s, invalid {sum(bool(r['invalid']) for r in c['requests'])}")


if __name__ == "__main__":
    for d in [Path(a) for a in sys.argv[1:]] or sorted(p for p in ROOT.iterdir() if p.is_dir()):
        report(d)
