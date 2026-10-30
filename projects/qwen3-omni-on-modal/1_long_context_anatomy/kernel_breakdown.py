"""Robustness check for analyze_trace.py: per-kernel exclusive time per decode step
under two overlap rules (earliest-started vs latest-started kernel owns overlaps),
on 40 steps sampled from steps 101-900. Usage: python kernel_breakdown.py <trace.json.gz>
"""
import collections, gzip, json, statistics as st, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from analyze_trace import category

ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
launch = {e["args"]["correlation"]: e for e in ev if e.get("cat") == "cuda_runtime" and e["name"] == "cudaGraphLaunch"}
byc = collections.defaultdict(list)
for e in ev:
    if e.get("cat") == "kernel":
        byc[e["args"].get("correlation")].append(e)
counts = collections.Counter(len(byc.get(c, [])) for c in launch)
step_kernels = max((n for n in counts if n >= 48), key=counts.get)  # same rule as analyze_trace.py
steps = [sorted(byc[c], key=lambda k: k["ts"]) for c, l in sorted(launch.items(), key=lambda kv: kv[1]["ts"])
         if len(byc.get(c, [])) == step_kernels]
steps = steps[100:900]  # middle 800 steps

def excl(ks, latest):
    out = collections.Counter()
    # sweep: collect boundaries, owner = earliest- or latest-started active kernel
    pts = sorted({k["ts"] for k in ks} | {k["ts"] + k["dur"] for k in ks})
    for a, b in zip(pts, pts[1:]):
        act = [k for k in ks if k["ts"] <= a and k["ts"] + k["dur"] >= b]
        if act:
            o = max(act, key=lambda k: k["ts"]) if latest else min(act, key=lambda k: k["ts"])
            out[o["name"][:70]] += b - a
    return out

for rule in (False, True):
    per = collections.defaultdict(list)
    for ks in steps[::20]:  # 40 sampled steps for speed
        c = excl(ks, rule)
        for n in set(k["name"][:70] for k in ks):
            per[n].append(c.get(n, 0))
    cat = collections.Counter()
    rows = sorted(((st.median(v), n) for n, v in per.items()), reverse=True)
    for m, n in rows:
        cat[category(n)] += m
    print(f"\n### owner = {'latest' if rule else 'earliest'}-started; per-step medians (ms), 40 steps")
    print({k: round(v / 1000, 3) for k, v in cat.items()})
    for m, n in rows[:14]:
        cnt = sum(1 for k in steps[0] if k["name"][:70] == n)
        print(f"{m/1000:7.3f} ms  x{cnt:3d}/step  {category(n):10s} {n}")
