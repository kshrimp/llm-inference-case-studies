"""Request-level numbers for experiment 1, from saved run records only.

Usage: python table.py [results/qwen3-omni-on-modal/1_anatomy/<run-id> ...]
(default: every anatomy run). Client clock only; trace numbers come from
analyze_trace.py.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "results/qwen3-omni-on-modal/1_anatomy"


def decode_gaps_ms(rec: dict) -> list[float]:
    """Gaps between consecutive streamed chunks after the first, in ms."""
    t = rec["chunk_offsets_us"]
    return [(b - a) / 1000 for a, b in zip(t, t[1:])]


def summarize(run_dir: Path) -> None:
    rec = json.loads((run_dir / "run.json").read_text())
    res = rec["result"]
    print(f"\n## {rec['run_id']}  status={res.get('status')}  error={res.get('error')}")
    print(f"GPU {res.get('gpu')}; server ready {res.get('server_ready_s')} s; "
          f"prompts { {k: v['prompt_tokens'] for k, v in (res.get('prompts') or {}).items()} }")
    measured = [s for s in res["steps"] if s.get("measured")]
    bad = [(s["step"], s["invalid"]) for s in measured if s["invalid"]]
    print(f"measured requests: {len(measured)}, invalid: {bad or 'none'}")
    by_kind: dict = {}
    for s in measured:
        if s["invalid"]:
            continue
        gaps = decode_gaps_ms(s)
        by_kind.setdefault(s["kind"], []).append({
            "ttft_ms": s["ttft_s"] * 1000, "completion_s": s["completion_s"],
            "chunks": len(s["chunk_offsets_us"]),
            "decode_ms_per_tok": (s["completion_s"] - s["ttft_s"]) * 1000 / (s["usage"]["completion_tokens"] - 1),
            "gaps": gaps,
        })
    print("| Prompt | n | TTFT ms, median [range] | Decode ms/token, median [range] | Completion s, median | Chunks |")
    print("| --- | ---: | --- | --- | ---: | ---: |")
    for kind, rows in sorted(by_kind.items()):
        tt = [r["ttft_ms"] for r in rows]
        dm = [r["decode_ms_per_tok"] for r in rows]
        print(f"| {kind} | {len(rows)} | {st.median(tt):.1f} [{min(tt):.1f}–{max(tt):.1f}] | "
              f"{st.median(dm):.3f} [{min(dm):.3f}–{max(dm):.3f}] | "
              f"{st.median(r['completion_s'] for r in rows):.3f} | "
              f"{st.median(r['chunks'] for r in rows)} |")
    # Context effect at matched step positions (client chunk gaps; one chunk per token if chunks == 1000).
    if "L" in by_kind and "S" in by_kind and all(r["chunks"] == 1000 for k in "LS" for r in by_kind[k]):
        n = 999
        med = {k: [st.median(r["gaps"][i] for r in by_kind[k]) for i in range(n)] for k in "LS"}
        for lo, hi in ((0, 100), (100, 500), (500, 999)):
            dl = st.median(med["L"][lo:hi])
            ds = st.median(med["S"][lo:hi])
            print(f"steps {lo + 1}-{hi}: L {dl:.3f} ms, S {ds:.3f} ms, L-S {dl - ds:+.3f} ms")
    else:
        print("matched-step comparison skipped: not one chunk per token")
    for s in res["steps"]:
        if s.get("torch_trace") or s.get("pyspy"):
            print(f"{s['step']}: invalid={s['invalid']} completion {s['completion_s']:.3f} s")


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or sorted(p for p in ROOT.iterdir() if p.is_dir())
    for d in dirs:
        summarize(d)


if __name__ == "__main__":
    main()
