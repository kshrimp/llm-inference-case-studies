"""Experiment 2b numbers from saved records. Usage: python table_2b.py"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "results/qwen3-omni-on-modal"


def main():
    for sub in ("2b_fp8_backends", "2b2_fp8_combined"):
        for run in sorted((ROOT / sub).iterdir()):
            res = json.loads((run / "run.json").read_text())["result"]
            print(f"\n## {run.name}")
            for name, s in res["sessions"].items():
                m = [q for q in s.get("requests", []) if q.get("measured")]
                line = f"{name}: ready {s.get('server_ready_s')} s"
                if m:
                    d = [q["decode_ms_per_tok"] for q in m]
                    t = [q["ttft_s"] * 1000 for q in m]
                    line += (f"; decode {st.median(d):.3f} [{min(d):.3f}-{max(d):.3f}] ms/token;"
                             f" TTFT {st.median(t):.1f} ms; invalid {[q['invalid'] for q in m if q['invalid']]}")
                if "needles" in s:
                    for cw in sorted({x["context_words"] for x in s["needles"]}):
                        q = [x for x in s["needles"] if x["context_words"] == cw]
                        lat = sorted(x["latency_s"] for x in q)
                        line += (f"; needles {cw}w {sum(x['correct'] for x in q)}/{len(q)}"
                                 f" (latency median {st.median(lat):.3f}, max {lat[-1]:.3f} s)")
                if "concurrent" in s:
                    c = s["concurrent"]
                    line += f"; c{c['n']} {c['output_tok_per_s']} tok/s over {c['wall_s']} s"
                print(line)


if __name__ == "__main__":
    main()
