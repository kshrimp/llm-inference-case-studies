"""Bandwidth roofline for Qwen3-Omni thinker decode on one H100 (batch 1).

Predicts the lowest possible ms/token from bytes read per decode step, so a
measured step can be split into "bandwidth floor" and "everything else"
(kernel inefficiency, launch gaps, CPU scheduling). Pure arithmetic from the
pinned config; no GPU needed. Usage: python roofline.py [--ctx 8000 --out 1000]
"""
import argparse

# Qwen/Qwen3-Omni-30B-A3B-Instruct @ 26291f79, thinker_config.text_config
L, H, NQ, NKV, D = 48, 2048, 32, 4, 128
EXPERTS, TOPK, FFN = 128, 8, 768
VOCAB = 152064
HBM_TBPS = 3.35  # H100 SXM HBM3 peak; Modal "H100!" is SXM


def active_params():
    attn = H * NQ * D + 2 * H * NKV * D + NQ * D * H
    moe = TOPK * 3 * H * FFN + H * EXPERTS  # gate/up/down of 8 experts + router
    return L * (attn + moe) + H * VOCAB  # lm_head; embedding is a gather


def kv_bytes_per_token(kv_bytes):
    return L * 2 * NKV * D * kv_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=8000)
    p.add_argument("--out", type=int, default=1000)
    a = p.parse_args()
    n = active_params()
    print(f"active params/token: {n / 1e9:.2f} B")
    for name, wb, kb in [("BF16 weights + BF16 KV", 2, 2), ("FP8 weights + BF16 KV", 1, 2),
                         ("FP8 weights + FP8 KV", 1, 1)]:
        mid_ctx = a.ctx + a.out / 2
        step_bytes = n * wb + kv_bytes_per_token(kb) * mid_ctx
        floor_ms = step_bytes / (HBM_TBPS * 1e12) * 1e3
        print(f"{name:24s} {step_bytes / 1e9:5.2f} GB/step  floor {floor_ms:.2f} ms/tok"
              f"  ({a.out} tok >= {floor_ms * a.out / 1e3:.2f} s)")
    print(f"KV per token: {kv_bytes_per_token(2) / 1024:.0f} KiB BF16; "
          f"{a.ctx + a.out} tokens = {kv_bytes_per_token(2) * (a.ctx + a.out) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
