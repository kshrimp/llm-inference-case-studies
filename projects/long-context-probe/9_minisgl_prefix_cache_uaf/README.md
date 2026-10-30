# Experiment 9: mini-SGLang prefix-cache use-after-free under pool pressure

**Question.** [Experiment 7b](../7_fp8_long_context_retrieval/FINDINGS.md) found deterministic, dtype-independent needle-retrieval failures that need **both** radix prefix-cache reuse **and** KV-pool pressure. Is this the suspected use-after-free in mini-SGLang's `CacheManager.cache_req`, and does a two-line remap fix it?

## Why we ran this

Experiment 7b showed that BF16's needle failures were deterministic, independent of dtype, and needed both prefix-cache reuse and pool pressure. That pattern does not fit numerical error; it fits KV pages being shared or overwritten. A silent wrong-answer bug in the prefix cache matters more than any FP8 result: it corrupts outputs with no error, and it only appears under load, as in production. It also explained experiment 7's misleading "FP8 is more accurate" result. Experiment 8 had ruled out an SGLang FP8 PR, so a confirmed fix here became the best candidate for an upstream contribution. It needed proof at three levels: a unit test, a runtime detector, and an end-to-end fix.

## What we did

1. Read `scheduler/cache.py` and wrote down the suspected mechanism below before running anything.
2. Wrote a CPU regression test in the style of the existing cache tests, plus a 7-line fix, in the clone `~/projects/upstream/mini-sglang-cache-fix`, and exported them as two patches.
3. **9-unit:** ran the test on CPU in Modal, first on base + test only (expected to fail), then on base + fix (expected to pass).
4. **9a detector:** reran the BF16 needle workload with an observe-only `_allocate` wrapper that flags any newly allocated page still referenced by a running request. Every experiment-7 failure turned out to be a victim.
5. **9b fix:** reran the BF16 needle workload on the fixed image: 84/90 → 90/90.

**Suspected mechanism** (source reading, pinned `9a91cfaf`, which is also mini-SGLang main at the pinned commit):
1. Several requests with the **same prefix** are prefilled in one batch. None had a cache hit at admission, so each wrote its own pages.
2. After prefill, `Scheduler` calls `cache_manager.cache_req(req, finished=False)` for each. The first inserts the prefix into the radix tree. For the next one, `insert_prefix` finds the prefix already present, so `cache_req` frees that request's duplicate pages, `[old_handle.cached_len, cached_len)`.
3. The request keeps decoding, but **its `page_table` row still points to the freed pages**. Nothing remaps it to the tree's copy.
4. Freed pages are appended to the end of `free_slots`, and `_allocate` takes from the front, so they are reused only when the pool runs low. Then another request overwrites them, and this request attends to someone else's KV. That explains why pool pressure is required and why outputs garble.

**Fix** ([`cache_req_uaf_fix.patch`](cache_req_uaf_fix.patch), sha256 `8bee25a2…`): for a still-running request, clone the duplicate slice, point the request's page-table entries at the tree's indices (`new_handle.get_matched_indices()`), then free the clone. This adds 7 lines in `scheduler/cache.py`, plus a CPU regression test, `tests/core/test_cache_req_shared_prefix.py`, in the style of the existing `tests/core/test_cache_allocate.py`. [`regression_test_only.patch`](regression_test_only.patch) adds only the test (sha256 `130c7007…`). The working clone is `~/projects/upstream/mini-sglang-cache-fix`, branch `fix-cache-req-uaf`.

**Runs** (runner: experiment 3's [app.py](../3_fp8_kv_cache_feasibility/app.py), sequential):
- **9-unit (CPU only):** run the regression test (plus the existing `test_cache_allocate.py`) on base + test only, where it should fail, and on base + fix, where it should pass.
- **9a detector (GPU):** BF16 needle, all 90 prompts, radix cache, with an `_allocate` wrapper that flags any newly allocated page still referenced by a running request's page table. It observes and does not change behavior. Expect events, with victims matching the failed prompts.
- **9b fix (GPU):** BF16 needle on the fixed image. Expect the 5–6 prefix-cache failures to disappear, with accuracy near the `naive` cache result (89/90).

**Expected signal (agent-written).** The test fails on base and passes with the fix. The detector reports reuse of live pages, with victims among the failed prompts. The fixed engine recovers the failures.

## How to run

Run from the repository root. `bash projects/long-context-probe/9_minisgl_prefix_cache_uaf/run.sh` runs everything below in order.

```bash
cd /path/to/llm-inference-case-studies
A=projects/long-context-probe/3_fp8_kv_cache_feasibility/app.py
uv run modal run $A::main --stage unittest --variant bf16_testonly # 9-unit: test only, expect FAIL
uv run modal run $A::main --stage unittest --variant bf16_cachefix # 9-unit: test + fix, expect PASS
uv run modal run $A::main --stage needle --variant bf16_uafdetect --repeat 1 # 9a: live-page reuse detector
uv run modal run $A::main --stage needle --variant bf16_cachefix --repeat 1 # 9b: fixed engine
uv run python projects/long-context-probe/9_minisgl_prefix_cache_uaf/table.py
uv run modal app list
```

To run the test against a local clone instead (CPU, no Modal): apply [`regression_test_only.patch`](regression_test_only.patch) or [`cache_req_uaf_fix.patch`](cache_req_uaf_fix.patch) to `mini-sglang@9a91cfaf` with `git apply`, then run `pytest tests/core/test_cache_req_shared_prefix.py tests/core/test_cache_allocate.py`.

Cost: 2 CPU containers (cents) + 2 GPU containers × $0.34 worst case. Results go to `results/long-context-probe/minisgl-prefix-cache-uaf/` (unit tests) and `results/long-context-probe/fp8-kv-needle/` (needle runs).

## Files

- [`cache_req_uaf_fix.patch`](cache_req_uaf_fix.patch), [`regression_test_only.patch`](regression_test_only.patch): fix + test, and test only.
- [`run.sh`](run.sh), [`table.py`](table.py): commands and analysis (unit-test outcomes, detector events vs failures, fixed-engine accuracy).
- Results and analysis: [FINDINGS.md](FINDINGS.md).
