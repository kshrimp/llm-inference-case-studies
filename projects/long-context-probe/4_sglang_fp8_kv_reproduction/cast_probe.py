"""Does torch's CUDA BF16 -> float8_e4m3fn cast saturate or produce NaN, by memory layout?

Mini-SGLang casts K as a strided slice of the fused QKV tensor; SGLang may pass a
contiguous tensor. Same torch 2.9.1 as both engines (SGLang 0.5.9 image).
"""

import json
from pathlib import Path

import modal

app = modal.App("fp8-cast-probe")
image = modal.Image.from_registry("lmsysorg/sglang:v0.5.9-cu129-amd64").entrypoint([])
VALUES = [440.0, 450.0, 460.0, 470.0, 516.0, -516.0, 0.01, 89.5]


@app.function(image=image, gpu="L4", cpu=2, memory=8192, max_containers=1,
              startup_timeout=300, timeout=300, scaledown_window=60)
def probe() -> dict:
    import torch

    f8 = torch.float8_e4m3fn
    n = len(VALUES)
    base = torch.tensor(VALUES, dtype=torch.bfloat16, device="cuda")
    rows = base.repeat(64, 1)  # [64, n] contiguous
    fused = torch.zeros(64, 4 * n, dtype=torch.bfloat16, device="cuda")
    fused[:, 2 * n:3 * n] = rows  # K-like slice of a fused QKV row: row stride 4n, inner stride 1
    strided = fused[:, 2 * n:3 * n]
    transposed = rows.t().contiguous().t()  # inner stride != 1
    cases = {
        "cpu_contiguous": base.cpu().to(f8),
        "cuda_contiguous_1d": base.to(f8),
        "cuda_contiguous_2d": rows.to(f8)[0],
        "cuda_row_strided_slice (mini-SGLang K)": strided.to(f8)[0],
        "cuda_transposed": transposed.to(f8)[0],
        "cuda_from_float32": base.float().to(f8),
        "cuda_clamp_then_cast": base.clamp(-448, 448).to(f8),
    }
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "input": VALUES, "strided_is_contiguous": strided.is_contiguous(),
            "results": {k: v.float().cpu().tolist() for k, v in cases.items()}}


@app.local_entrypoint()
def main():
    result = probe.remote()
    out = Path("results/long-context-probe/sglang-fp8-kv-reproduction/cast_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {out}")
    print(json.dumps(result, indent=2))
