"""Write SHA256SUMS and validation.json for every saved run of this project (local).

Pattern from projects/omni-pipeline-probe/validate.py. For each run folder under
results/qwen3-omni-on-modal/<step>/<run_id>/ it records a checksum of every artifact
and run-level checks: run status, request validity counts (every request record in
the run that carries an `invalid` list), and pin fields. Experiment 3 records are
single JSON files per cold run and are checked per file. Raw artifacts are never
modified; SHA256SUMS and validation.json are excluded from their own checksums.

Usage: uv run python projects/qwen3-omni-on-modal/validate.py [--check]
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("results/qwen3-omni-on-modal")
GENERATED = {"SHA256SUMS", "validation.json"}
CONTRACT_GAPS = {
    "manifest.json": "run.json holds pins (omni_sha, base_image), stage, timing and status; "
                     "measurement verdicts are recorded here and spend in the study-plan cost log",
    "config.resolved.json": "server commands are in run.json; the resolved config is saved for "
                            "experiment 1 (resolved_config) and checked on CPU in step 0",
    "workload.jsonl": "prompts are rebuilt from the pinned text (SHA-256 checked in-container); "
                      "their hashes and token counts are in run.json",
    "resources.jsonl": "GPU memory samples or peaks are inside run.json without a separate record",
    "actual_cost_per_run": "Modal bills per app; posted app totals are in the study-plan cost log",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def requests_in(obj, found: list) -> list:
    """Every dict with an `invalid` list: a client request record."""
    if isinstance(obj, dict):
        if isinstance(obj.get("invalid"), list) and "status" in obj:
            found.append(obj)
        for v in obj.values():
            requests_in(v, found)
    elif isinstance(obj, list):
        for v in obj:
            requests_in(v, found)
    return found


def checks(record: dict) -> dict:
    res = record.get("result", record)
    reqs = requests_in(res, [])
    return {"status": res.get("status"), "error": res.get("error"),
            "omni_sha": record.get("omni_sha"), "requests": len(reqs),
            "requests_invalid": sum(bool(r["invalid"]) for r in reqs)}


def main(check: bool) -> int:
    bad = 0
    for run_dir in sorted(p for p in ROOT.glob("*/*") if p.is_dir() and (p / "run.json").exists()):
        files = sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name not in GENERATED)
        sums = {str(p.relative_to(run_dir)): sha256(p) for p in files}
        if check:
            stored = dict(reversed(l.split("  ", 1)) for l in (run_dir / "SHA256SUMS").read_text().splitlines())
            diff = sorted(k for k in set(stored) | set(sums) if stored.get(k) != sums.get(k))
            bad += bool(diff)
            print(f"{run_dir}: {'OK' if not diff else f'MISMATCH {diff[:5]}'}")
            continue
        (run_dir / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
        v = checks(json.loads((run_dir / "run.json").read_text()))
        v.update(schema_version=1, artifacts=len(sums), contract_gaps=CONTRACT_GAPS)
        (run_dir / "validation.json").write_text(json.dumps(v, indent=2))
        print(f"{run_dir}: {v['status']} requests={v['requests']} invalid={v['requests_invalid']}")
    cold = ROOT / "3_cold_start"
    if cold.exists() and not check:
        rows = {}
        for f in sorted(cold.glob("*.json")):
            if f.name in GENERATED:
                continue
            r = json.loads(f.read_text())
            c = r.get("container") or {}
            rows[f.name] = {"sha256": sha256(f), "condition": r.get("condition"), "model": r.get("model"),
                            "first_token_s": r["client"].get("first_token_s"),
                            "error": r["client"].get("error") or c.get("error"),
                            "completion_tokens": (c.get("usage") or {}).get("completion_tokens"),
                            "text_sha256": c.get("text_sha256")}
        (cold / "validation.json").write_text(json.dumps({"schema_version": 1, "runs": rows}, indent=2))
        print(f"{cold}: {len(rows)} cold-start records")
    return bad


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
