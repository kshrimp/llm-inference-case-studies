"""Experiment 2 numbers from saved records only. Usage: python table.py [run_dir ...]
(default: every 2a smoke run)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "results/qwen3-omni-on-modal/2a_fp8_smoke"


def report(run_dir: Path) -> None:
    res = json.loads((run_dir / "run.json").read_text())["result"]
    probe = res.get("gpu_probe", {})
    print(f"\n## {run_dir.name}  GPU {res.get('gpu')}")
    print(f"CUDA cast {probe.get('cast_in')} -> {probe.get('cast_out_cuda')}; "
          f"device copy {probe.get('copy_gb_per_s', 0):.0f} GB/s (read + write)")
    print("| Cond | Ready s | Weight load s | KV dtype | KV tokens | TTFT ms (8k) | Decode ms/token (8k/1k, n=1) | Valid |")
    print("| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |")
    for name, c in res["conditions"].items():
        log = c.get("server_log", "")
        load = re.search(r"Load weight end\. elapsed=([\d.]+) s", log)
        kv = re.search(r"KV Cache is allocated\. dtype: (\S+), #tokens: (\d+)", log)
        f = c["steps"].get("full_L", {})
        print(f"| {name} | {c.get('server_ready_s')} | {load.group(1) if load else None} | "
              f"{kv.group(1).rstrip(',') if kv else None} | {kv.group(2) if kv else None} | "
              f"{(f.get('ttft_s') or 0) * 1000:.1f} | {f.get('decode_ms_per_tok', 0):.3f} | "
              f"{'yes' if f.get('invalid') == [] else f.get('invalid')} |")
    for name, g in (res.get("gates") or {}).items():
        cells = ", ".join(f"{k}: first divergence {v['first_divergence']}/{v['ref_len']}"
                          for k, v in g.items() if isinstance(v, dict))
        print(f"gate {name}: pass={g['pass']} reasons={g['reasons']}; {cells}")


if __name__ == "__main__":
    for d in [Path(a) for a in sys.argv[1:]] or sorted(p for p in ROOT.iterdir() if p.is_dir()):
        report(d)
