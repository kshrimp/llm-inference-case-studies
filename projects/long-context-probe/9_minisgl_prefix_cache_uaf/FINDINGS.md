# Experiment 9 findings: a use-after-free in mini-SGLang's prefix cache, confirmed and fixed (2026-10-25)

**Result: confirmed.** In mini-SGLang (pinned `9a91cfaf`, which is also its main branch as of 2026-10-25), `CacheManager.cache_req(req, finished=False)` frees a still-running request's duplicate prefix pages without remapping its page table. Under KV-pool pressure those pages are reallocated to other requests, so the original request attends to someone else's KV. Three independent checks agree:

1. A CPU regression test **fails on the current code and passes with the fix**.
2. An allocation detector shows that **every needle failure in experiment 7 was a victim** of live-page reallocation.
3. With a 7-line fix, the same BF16 workload goes **84/90 → 90/90**.

This is an engine correctness bug, independent of FP8. It explains experiment 7's "FP8 is more accurate" artifact.

Reproduce with `uv run python projects/long-context-probe/9_minisgl_prefix_cache_uaf/table.py`.

## 1. Mechanism (source, `python/minisgl/scheduler/cache.py`)

```python
cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
self.unlock(old_handle)
# this part is already in the prefix cache, free it
self._free(page_indices[old_handle.cached_len : cached_len])   # <- request's own pages
...
else:  # keep the tail part, update the handle
    req.cache_handle = new_handle; self.lock(new_handle)       # <- page_table row not remapped
```

The trigger: several requests share a prefix, are admitted before any of them is cached (so none gets a prefix hit), and are prefilled together. `Scheduler` calls `cache_req(..., finished=False)` for each after prefill. The first request inserts the prefix. For each later one, `insert_prefix` reports the prefix as already present (`cached_len > old_handle.cached_len`), so `cache_req` frees that request's own pages for `[old_handle.cached_len, cached_len)`. The request keeps decoding with its page-table row still pointing at them.

Why pool pressure is required: `_free` appends to the **end** of `free_slots` and `_allocate` takes from the **front**, so freed pages are reused only once the pool runs low. Until then the stale pages still hold the request's own correct KV.

## 2. Regression test (CPU, `tests/core/test_cache_req_shared_prefix.py`)

Two requests with the same 8-token prompt (page size 4) are admitted and prefilled, then cached with `finished=False`. The test asserts that neither page-table row references a freed page, that the second row now points at the first request's cached pages, and that no page leaked (6 of 8 free).

| Image | Result | Run ID |
| --- | --- | --- |
| base + test only (`regression_test_only.patch`) | **1 failed, 6 passed**: `AssertionError: running request 1 references freed pages {8, 12}` | `20261025T041120Z-unittest-bf16_testonly` |
| base + fix + test (`cache_req_uaf_fix.patch`) | **7 passed** (the new test + the 6 existing `test_cache_allocate.py` tests) | `20261025T041147Z-unittest-bf16_cachefix` |

The first pair of unit-test runs (`20261025T040755Z`, `20261025T040807Z`) exited with pytest usage code 4, because the project's `addopts` require `pytest-cov`. They were rerun with `-o addopts=`; nothing was executed in the first pair.

## 3. Detector run (BF16, radix cache, all 90 needle prompts, `20261025T040816Z-needle-bf16_uafdetect-r1`)

A wrapper on `CacheManager._allocate` flags each newly allocated page that is still referenced by a running request's page-table row, up to its `device_len`. It observes and does not change behavior; accuracy stayed 84/90, as in the baseline.

- **20 events, 63 pages, 18 victim requests:** 13 of the ~2k-token prompts and 5 of the ~8k ones.
- **All 6 failed prompts are victims**: 2048@0.5 #3 #7, 2048@0.9 #1 #2 #3, 8000@0.9 #1. **No failure occurred outside the victims.**
- **12 victims still answered correctly.** Corruption breaks the answer only when the overwritten pages matter for retrieval. Failures cluster at larger losses: 6 pages each for 2048@0.9 #1–#3 and 10 pages for 8000@0.9 #1. Most correct victims lost 1–3 pages; the two 0.5-depth failures lost 3–4.

## 4. Fixed engine (BF16, radix cache, all 90 prompts, `20261025T041553Z-needle-bf16_cachefix-r1`)

| Run | Prefix cache | Pool tokens | Correct |
| --- | --- | ---: | ---: |
| base (experiment 7, r1 and r2) | radix | 172,544 | 84/90 (identical across repeats) |
| base, prefix cache disabled | naive | 172,544 | 89/90 |
| **fixed** | radix | 172,544 | **90/90** |

All 6 baseline failures are recovered, and its text matches the base run on 76/90 prompts. The fix changes only the victims' outputs and their downstream batching.

## 5. The fix (`cache_req_uaf_fix.patch`, sha256 `8bee25a2…`)

```python
duplicate = page_indices[old_handle.cached_len : cached_len].clone()
if not finished and len(duplicate) > 0:
    shared = new_handle.get_matched_indices()[old_handle.cached_len : cached_len]
    page_indices[old_handle.cached_len : cached_len] = shared   # point at the cached copy
self._free(duplicate)
```

`page_indices` is a **view** of the page table, so the duplicate slice is cloned before the remap. Remapping first would free the tree's pages instead. Finished requests are unchanged. With the naive cache, `insert_prefix` never reports a shared prefix, so the branch never runs.

## Implications

- **mini-SGLang users:** any workload with concurrent shared-prefix requests (system prompts, few-shot templates, RAG with shared context) **and** a nearly full KV pool can silently produce corrupted outputs. It is deterministic for a given workload and scheduler order.
- **For this project:** experiment 7's FP8 "advantage" and BF16's 6 misses were this bug. FP8 hid it by doubling the pool, which reduced reuse of freed pages.
- **Contribution direction (updated 2026-10-25):** the user will not open a mini-SGLang PR. The fix and regression test remain local study artifacts in `~/projects/upstream/mini-sglang-cache-fix`, branch `fix-cache-req-uaf`. A full-SGLang contribution would require an independent check of its equivalent path, reproduction on a current pin, and a justified fix. Nothing has been posted.

## Limits

- One workload (the needle prompts), one model and GPU. The detector checks only decoding requests' rows, so prefill-time victims would be missed.
- The page-count vs failure relationship is descriptive. Which overwritten positions matter for retrieval was not mapped.

## Cost

Two CPU unit-test containers (cents, rerun once) and two GPU needle runs; actuals are on the study-plan status board.
