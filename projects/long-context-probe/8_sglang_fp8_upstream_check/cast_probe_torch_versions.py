"""Does torch's CUDA BF16 -> float8_e4m3fn cast saturate, by torch version?

CPU bisect (2026-10-24): 2.10.0 and 2.12.0 give NaN for 470/516; 2.13.0, 2.14.0, and 2.14.1 give 448.
SGLang 0.5.9 ships torch 2.9.1 (NaN, see cast_probe.json); SGLang main pins torch 2.14.1.
This checks the CUDA path on both sides of that change.
"""

import json
from pathlib import Path

import modal

VERSIONS = {"2.12.0": "torch==2.12.0", "2.14.1": "torch==2.14.1"}
VALUES = [440.0, 450.0, 460.0, 464.0, 470.0, 516.0, -516.0]
app = modal.App("fp8-cast-probe-torch-versions")
images = {v: modal.Image.debian_slim(python_version="3.12").pip_install(spec) for v, spec in VERSIONS.items()}
LIMITS = dict(gpu="L4", cpu=2, memory=8192, max_containers=1,
              startup_timeout=300, timeout=300, scaledown_window=60)


def probe() -> dict:
    import torch

    f8 = torch.float8_e4m3fn
    x = torch.tensor(VALUES, dtype=torch.bfloat16)
    wide = torch.zeros(4, 3 * len(VALUES), dtype=torch.bfloat16, device="cuda")
    wide[:, len(VALUES):2 * len(VALUES)] = x.cuda()
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "input": VALUES,
            "cpu": x.to(f8).float().tolist(),
            "cuda_contiguous": x.cuda().to(f8).float().cpu().tolist(),
            "cuda_strided_slice": wide[:, len(VALUES):2 * len(VALUES)].to(f8)[0].float().cpu().tolist(),
            "cuda_from_float32": x.float().cuda().to(f8).float().cpu().tolist()}


@app.function(image=images["2.12.0"], **LIMITS)
def probe_212() -> dict:
    return probe()


@app.function(image=images["2.14.1"], **LIMITS)
def probe_2141() -> dict:
    return probe()


@app.local_entrypoint()
def main():
    results = [probe_212.remote(), probe_2141.remote()]  # sequential: one GPU container at a time
    out = Path("results/long-context-probe/sglang-fp8-kv-reproduction/cast_probe_torch_versions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {out}")
    print(json.dumps(results, indent=2))
