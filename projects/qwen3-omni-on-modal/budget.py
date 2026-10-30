"""Worst-case Modal cost ceilings for the Qwen3-Omni on Modal plan.

Every ceiling is containers x (startup timeout + run timeout + idle) x per-second
price, so a reader can recompute it. Prices are the ones the omni probe used
(read 2026-10-25); re-read modal.com/pricing before the first GPU launch and
update PRICE_PER_S. Usage: python budget.py
"""

PRICE_PER_S = {"H100": 0.001097, "cpu_core": 0.0000131, "mem_gib": 0.00000222}
GPU_SHAPE = {"cpu": 8, "mem_gib": 64}
CPU_SHAPE = {"cpu": 4, "mem_gib": 8}

APPROVED = 50.00
SPENT_BEFORE = 4.647829  # account total after omni exp 3 container C', 2026-10-27
SPENT_PROJECT = 1.52217699  # steps 0 + exp 1 app rows, billing --json 2026-10-28
ACCOUNT_AFTER_4A = 10.13902963  # billing --json from 2026-10-01, rechecked 2026-10-29 before 2c


def per_hour(gpu: bool) -> float:
    shape = GPU_SHAPE if gpu else CPU_SHAPE
    s = shape["cpu"] * PRICE_PER_S["cpu_core"] + shape["mem_gib"] * PRICE_PER_S["mem_gib"]
    return 3600 * (s + (PRICE_PER_S["H100"] if gpu else 0))


# (name, gpu, containers, startup_min, run_min, idle_min); the minutes are the
# timeouts the runner enforces, so a container cannot bill past them.
PLAN = [
    ("0. CPU image build + FP8 weight download", False, 2, 5, 60, 1),
    ("1. Anatomy: 8k/1k + 1k/1k control, traces", True, 2, 10, 30, 2),
    ("2a. FP8 smoke: B, C, A reference (3 server starts)", True, 1, 10, 30, 2),
    ("2b-1. FP8 backend sessions + traces (redesign)", True, 1, 10, 30, 2),
    ("2b-2. Repeat of winning config + quality set", True, 1, 10, 30, 2),
    ("3. Cold start: FP8 3 conditions x 3 + BF16 x 2", True, 11, 10, 1, 1),
    ("3s. Snapshot creation container", True, 1, 10, 10, 2),
    ("4a. Memory fraction + closed-loop scan", True, 1, 10, 30, 2),
    ("2c. Triton FP8 dense tuning + one session", True, 1, 10, 30, 2),  # added 2026-10-29, user OK
    ("4b. Capacity: BF16 + FP8 open-loop sweep", True, 1, 10, 80, 2),  # 87 -> 92 min, user OK 2026-10-29
    ("Reserve: one retry of a 42-min GPU container", True, 1, 10, 30, 2),
]


def main():
    print(f"GPU container: ${per_hour(True):.3f}/h   CPU container: ${per_hour(False):.3f}/h")
    total = 0.0
    for name, gpu, n, start, run, idle in PLAN:
        minutes = n * (start + run + idle)
        cost = minutes / 60 * per_hour(gpu)
        total += cost
        print(f"{name:48s} {n:2d} x {start + run + idle:3d} min = {minutes:4d} min  ${cost:6.2f}")
    print(f"{'Plan worst case':48s} {'':24s} ${total:6.2f}")
    print(f"Spent before: ${SPENT_BEFORE:.2f}; worst-case account total ${SPENT_BEFORE + total:.2f}"
          f" of ${APPROVED:.2f} (headroom ${APPROVED - SPENT_BEFORE - total:.2f})")
    done = sum(n * (s + r + i) / 60 * per_hour(g) for name, g, n, s, r, i in PLAN[:2])
    left = total - done
    print(f"Spent on this project so far: ${SPENT_PROJECT:.2f}; remaining rows (2a onward) ${left:.2f};"
          f" worst-case account total ${SPENT_BEFORE + SPENT_PROJECT + left:.2f} of ${APPROVED:.2f}")
    todo = [r for r in PLAN if r[0].split(".")[0] in ("2c", "3", "3s", "4b", "Reserve: one retry of a 42-min GPU container")]
    todo_cost = sum(n * (s + r + i) / 60 * per_hour(g) for name, g, n, s, r, i in todo)
    print(f"After 4a: billed ${ACCOUNT_AFTER_4A:.2f}; rows not yet run (2c, 3, 3s, 4b, reserve) ${todo_cost:.2f};"
          f" worst-case account total ${ACCOUNT_AFTER_4A + todo_cost:.2f} of ${APPROVED:.2f}")


if __name__ == "__main__":
    main()
