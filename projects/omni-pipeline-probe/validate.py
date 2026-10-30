"""Write SHA256SUMS and validation.json for every saved run of the omni probe (local).

For each run folder under results/omni-pipeline-probe/{setup,smoke,measure}/ this
records a checksum of every artifact, and for measurement runs the per-request
verdicts from analysis.load_rows plus run-level checks. It also lists, as nulls
with reasons, the docs/PROJECT_REFERENCE.md data-contract artifacts this layout
does not have. Raw artifacts are never modified. Re-running is idempotent:
SHA256SUMS and validation.json are excluded from their own checksums.

Usage: uv run python projects/omni-pipeline-probe/validate.py [--check]
  --check  verify existing SHA256SUMS instead of writing them
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import load_rows  # noqa: E402

ROOT = Path("results/omni-pipeline-probe")
GENERATED = {"SHA256SUMS", "validation.json"}
# Data-contract artifacts (PROJECT_REFERENCE.md "Data contracts") and why this probe lacks them.
CONTRACT_GAPS = {
    "manifest.json": "run.json holds pins, limits, GPU, timing and status, but no separate "
                     "measurement verdict field, artifact checksums or cleanup outcome; those are "
                     "recorded here and in the study-plan cost log",
    "config.resolved.json": "requested settings are in run.json; the resolved pipeline config "
                            "was checked once on CPU (setup cpu-check) but not saved per run",
    "workload.jsonl": "the workload is the fixed step list in run.json plus inputs_manifest.json "
                      "(hashes of every WAV and the text passage)",
    "resources.jsonl": "GPU memory/utilization samples are inside run.json (gpu_samples), "
                       "without a separate unit/source/scope record",
    "summary.json": "summaries are produced by the experiment table.py scripts, not stored",
    "actual_cost_per_run": "Modal billing is per app; each run used its own app, and the "
                           "posted app totals are in the study-plan cost log",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifacts(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name not in GENERATED)


def write_sums(run_dir: Path) -> dict:
    sums = {str(p.relative_to(run_dir)): sha256(p) for p in artifacts(run_dir)}
    (run_dir / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    return sums


def check_sums(run_dir: Path) -> list[str]:
    path = run_dir / "SHA256SUMS"
    if not path.exists():
        return [f"{run_dir}: no SHA256SUMS"]
    problems = []
    listed = {}
    for line in path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
        target = run_dir / name
        if not target.is_file() or sha256(target) != digest:
            problems.append(f"{run_dir / name}: missing or changed")
    for p in artifacts(run_dir):
        if str(p.relative_to(run_dir)) not in listed:
            problems.append(f"{p}: not in SHA256SUMS")
    return problems


def validate_measure(run_dir: Path) -> dict:
    info, rows = load_rows(run_dir)
    ids = [json.loads(l)["request_id"] for l in
           (run_dir / "requests.jsonl").read_text().splitlines() if l.strip()]
    measured = [r for r in rows if r["measured"]]
    return {
        "run": info,
        "measured_requests": len(measured),
        "valid_requests": sum(r["valid"] for r in measured),
        "unmeasured_requests": [r["step"] for r in rows if not r["measured"]],
        "request_ids_unique": len(set(ids)) == len(ids),
        "verdict": "valid" if measured and all(r["valid"] for r in measured) else "invalid",
        "requests": [{"step": r["step"], "valid": r["valid"], "reasons": r["invalid_reasons"]}
                     for r in measured],
    }


def main() -> None:
    check_only = "--check" in sys.argv[1:]
    run_dirs = sorted(d for kind in ("setup", "smoke", "measure")
                      for d in (ROOT / kind).glob("*") if (d / "run.json").exists())
    if check_only:
        problems = [p for d in run_dirs for p in check_sums(d)]
        print("\n".join(problems) or f"all {len(run_dirs)} runs match their SHA256SUMS")
        sys.exit(1 if problems else 0)
    for run_dir in run_dirs:
        sums = write_sums(run_dir)
        run = json.loads((run_dir / "run.json").read_text())
        record = {"schema_version": 1, "run_id": run["run_id"], "stage": run["stage"],
                  "execution_status": run["status"], "artifact_sha256": sums,
                  "data_contract_gaps": CONTRACT_GAPS}
        if run["stage"] == "measure":
            record.update(validate_measure(run_dir))
        elif run["stage"] == "smoke":
            checks = run_dir / "checks.json"
            gate = json.loads(checks.read_text())["gate_for_container_B"] if checks.exists() else None
            record["verdict"] = ("valid" if gate and gate["pass"] else "invalid") if gate else None
            record["verdict_scope"] = "compatibility gate only (smoke_report.py), not a measurement"
        else:
            record["verdict"] = None
            record["verdict_scope"] = "setup step; see run.json result"
        (run_dir / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{run_dir.name}: {len(sums)} artifacts, verdict {record.get('verdict')}"
              + (f", {record['valid_requests']}/{record['measured_requests']} valid"
                 if run["stage"] == "measure" else ""))


if __name__ == "__main__":
    main()
