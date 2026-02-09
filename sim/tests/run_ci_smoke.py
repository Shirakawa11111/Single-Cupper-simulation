"""
Run a minimal CI smoke bundle and write one aggregated summary JSON.

Bundle tasks:
1) `phase2_quick`: fast phase-2 gate (quick crack-scan config).
2) `multi_align_smoke`: fixture-based multi-condition alignment smoke.
3) `seed_ci_smoke`: fixture-based seed CI summarization smoke.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any


def _default_out_root() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"ci_smoke_{ts}"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _run(name: str, cmd: list[str], logs_dir: Path) -> dict[str, Any]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = perf_counter() - t0
    stdout_path = logs_dir / f"{name}.stdout"
    stderr_path = logs_dir / f"{name}.stderr"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return {
        "name": name,
        "command": cmd,
        "returncode": int(proc.returncode),
        "duration_s": dt,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "runner_passed": proc.returncode == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run minimal CI smoke bundle.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-multi-align", action="store_true")
    parser.add_argument("--skip-seed-ci", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root or _default_out_root()
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now()
    records: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    py = args.python

    if not args.skip_phase2:
        phase2_out = out_root / "phase2_quick"
        phase2_cmd = [
            py,
            "sim/tests/regress_phase2.py",
            "--out",
            str(phase2_out),
            "--skip-phase1-suite",
            "--scan-config",
            "sim/configs/crack_onset_scan_quick.yaml",
            "--scan-max-cases",
            "1",
            "--scan-min-onset-cases",
            "0",
            "--max-runtime-warnings",
            "200",
            "--max-crack-cg-nonconverged-steps",
            "12",
            "--max-nonfinite-count",
            "0",
        ]
        rec = _run("phase2_quick", phase2_cmd, logs_dir)
        summary_path = phase2_out / "summary.json"
        summary = _read_json(summary_path)
        task_passed = bool(rec["runner_passed"] and isinstance(summary, dict) and summary.get("passed", False))
        rec["summary_json"] = str(summary_path)
        rec["summary"] = summary
        rec["passed"] = task_passed
        records.append(rec)
        if not task_passed:
            failure_reasons.append("phase2_quick_failed")

    if not args.skip_multi_align:
        multi_out = out_root / "multi_align_smoke" / "summary.json"
        multi_cmd = [
            py,
            "sim/tests/regress_exp_alignment_multi.py",
            "--config",
            "sim/configs/exp_alignment_multi_ci_smoke.yaml",
            "--out",
            str(multi_out),
        ]
        rec = _run("multi_align_smoke", multi_cmd, logs_dir)
        summary = _read_json(multi_out)
        task_passed = bool(rec["runner_passed"] and isinstance(summary, dict) and summary.get("passed", False))
        rec["summary_json"] = str(multi_out)
        rec["summary"] = summary
        rec["passed"] = task_passed
        records.append(rec)
        if not task_passed:
            failure_reasons.append("multi_align_smoke_failed")

    if not args.skip_seed_ci:
        seed_out_dir = out_root / "seed_ci_smoke"
        seed_summary_json = seed_out_dir / "summary.json"
        seed_summary_md = seed_out_dir / "summary.md"
        seed_aggregate_csv = seed_out_dir / "aggregate.csv"
        seed_cmd = [
            py,
            "sim/tests/summarize_seed_robustness_ci.py",
            "--batch-dirs",
            "sim/tests/fixtures/seed_ci_smoke/batch_a,sim/tests/fixtures/seed_ci_smoke/batch_b",
            "--out",
            str(seed_summary_json),
            "--markdown-out",
            str(seed_summary_md),
            "--aggregate-csv-out",
            str(seed_aggregate_csv),
            "--confidence",
            "0.95",
        ]
        rec = _run("seed_ci_smoke", seed_cmd, logs_dir)
        summary = _read_json(seed_summary_json)
        task_passed = bool(
            rec["runner_passed"]
            and isinstance(summary, dict)
            and summary.get("all_seed_gate_passed", False)
        )
        rec["summary_json"] = str(seed_summary_json)
        rec["summary_markdown"] = str(seed_summary_md)
        rec["aggregate_csv"] = str(seed_aggregate_csv)
        rec["summary"] = summary
        rec["passed"] = task_passed
        records.append(rec)
        if not task_passed:
            failure_reasons.append("seed_ci_smoke_failed")

    finished = datetime.now()
    summary_payload = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "duration_s": (finished - started).total_seconds(),
        "out_root": str(out_root),
        "tasks": records,
        "failure_reasons": failure_reasons,
        "passed": len(records) > 0 and all(bool(r.get("passed", False)) for r in records),
    }

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(summary_path)
    print(json.dumps(summary_payload, indent=2))
    return 0 if summary_payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
