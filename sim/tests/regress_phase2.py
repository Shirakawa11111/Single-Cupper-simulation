"""
Phase-2 gate: stability checks + crack-onset screening.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any


def _default_out_dir() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"phase2_gate_{ts}"


@dataclass
class TaskResult:
    name: str
    command: list[str]
    returncode: int
    duration_s: float
    passed: bool
    summary_json: str
    runtime_warning_count: int
    failures: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _run_task(name: str, cmd: list[str], summary_json: Path) -> TaskResult:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dur = perf_counter() - t0

    (summary_json.parent / f"{name}.stdout").write_text(proc.stdout, encoding="utf-8")
    (summary_json.parent / f"{name}.stderr").write_text(proc.stderr, encoding="utf-8")

    report = _read_json(summary_json)
    report_passed = bool(report is not None and report.get("passed", True))
    passed = proc.returncode == 0 and report_passed
    runtime_warning_count = proc.stdout.count("RuntimeWarning") + proc.stderr.count("RuntimeWarning")
    failures: dict[str, Any] = {}
    if report is not None:
        report_warn = report.get("runtime_warning_count")
        if isinstance(report_warn, int):
            runtime_warning_count = max(runtime_warning_count, int(report_warn))
        report_fail = report.get("failure_reasons")
        if isinstance(report_fail, list) and report_fail:
            failures["failure_reasons"] = report_fail
        elif isinstance(report.get("failures"), dict):
            failures = report.get("failures", {})
    if proc.returncode != 0 and not failures:
        failures["runner"] = f"returncode={proc.returncode}"

    return TaskResult(
        name=name,
        command=cmd,
        returncode=proc.returncode,
        duration_s=dur,
        passed=passed,
        summary_json=str(summary_json),
        runtime_warning_count=runtime_warning_count,
        failures=failures,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase-2 regression gate.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for logs and summary.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable path.")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode for Phase-1 suite.")
    parser.add_argument(
        "--max-runtime-warnings",
        type=int,
        default=20,
        help="RuntimeWarning threshold for scripts that support this option.",
    )
    parser.add_argument(
        "--max-mechanical-cg-failures",
        type=int,
        default=None,
        help="Threshold for stability_diagnostics.mechanical_cg_failures.",
    )
    parser.add_argument(
        "--max-crack-cg-nonconverged-steps",
        type=int,
        default=0,
        help="Threshold for stability_diagnostics.crack_cg_nonconverged_steps.",
    )
    parser.add_argument(
        "--max-mechanical-not-accepted-steps",
        type=int,
        default=None,
        help="Threshold for stability_diagnostics.mechanical_not_accepted_steps.",
    )
    parser.add_argument(
        "--max-nonfinite-count",
        type=int,
        default=0,
        help="Threshold for stability_diagnostics.nonfinite_count.",
    )
    parser.add_argument(
        "--scan-config",
        type=Path,
        default=Path("sim/configs/crack_onset_scan.yaml"),
        help="Crack-onset scan config path.",
    )
    parser.add_argument("--scan-max-cases", type=int, default=None, help="Optional cap for scan cases.")
    parser.add_argument(
        "--scan-min-onset-cases",
        type=int,
        default=None,
        help="Override min onset-case count for scan gate.",
    )
    parser.add_argument(
        "--scan-min-notch-cycles-completed",
        type=int,
        default=None,
        help="Override notch-case minimum cycles_completed for scan gate.",
    )
    parser.add_argument(
        "--with-exp-alignment",
        action="store_true",
        help="Run experiment-alignment regression gate as part of phase-2.",
    )
    parser.add_argument(
        "--exp-alignment-config",
        type=Path,
        default=Path("sim/configs/fatigue_lowamp_align_locked_v4.yaml"),
        help="Config path for experiment-alignment gate.",
    )
    parser.add_argument(
        "--exp-alignment-rmse-tau-max",
        type=float,
        default=30.0,
        help="RMSE tau threshold passed to regress_exp_alignment.py.",
    )
    parser.add_argument(
        "--exp-alignment-mae-tau-max",
        type=float,
        default=25.0,
        help="MAE tau threshold passed to regress_exp_alignment.py.",
    )
    parser.add_argument(
        "--exp-alignment-rmse-gamma-max",
        type=float,
        default=4.2e-3,
        help="RMSE gamma threshold passed to regress_exp_alignment.py.",
    )
    parser.add_argument("--skip-phase1-suite", action="store_true")
    args = parser.parse_args()

    out_dir = args.out or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    py = args.python
    tasks: list[tuple[str, list[str], Path]] = []

    if not args.skip_phase1_suite:
        phase1_out = out_dir / "phase1_suite"
        cmd = [
            py,
            "sim/tests/run_phase1_suite.py",
            "--out",
            str(phase1_out),
            "--max-runtime-warnings",
            str(args.max_runtime_warnings),
        ]
        if args.strict:
            cmd.append("--strict")
        tasks.append(("phase1_suite", cmd, phase1_out / "summary.json"))

    config_out = out_dir / "config_runs"
    tasks.append(
        (
            "monotonic_config",
            [
                py,
                "sim/tests/run_virtual_cycle_config.py",
                "--config",
                "sim/configs/monotonic_baseline.yaml",
                "--summary-output",
                str(config_out / "monotonic_baseline_run_summary.json"),
                "--max-runtime-warnings",
                str(args.max_runtime_warnings),
                "--max-nonfinite-count",
                str(args.max_nonfinite_count),
            ],
            config_out / "monotonic_baseline_run_summary.json",
        )
    )
    if args.max_mechanical_cg_failures is not None:
        tasks[-1][1].extend(["--max-mechanical-cg-failures", str(args.max_mechanical_cg_failures)])
    if args.max_mechanical_not_accepted_steps is not None:
        tasks[-1][1].extend(["--max-mechanical-not-accepted-steps", str(args.max_mechanical_not_accepted_steps)])
    if args.max_crack_cg_nonconverged_steps is not None:
        tasks[-1][1].extend(
            ["--max-crack-cg-nonconverged-steps", str(args.max_crack_cg_nonconverged_steps)]
        )

    scan_out = out_dir / "crack_onset_scan"
    scan_cmd = [
        py,
        "sim/tests/scan_crack_onset.py",
        "--config",
        str(args.scan_config),
        "--out",
        str(scan_out),
        "--max-runtime-warnings",
        str(args.max_runtime_warnings),
        "--max-nonfinite-count",
        str(args.max_nonfinite_count),
    ]
    if args.scan_max_cases is not None:
        scan_cmd.extend(["--max-cases", str(args.scan_max_cases)])
    if args.scan_min_onset_cases is not None:
        scan_cmd.extend(["--min-onset-cases", str(args.scan_min_onset_cases)])
    if args.scan_min_notch_cycles_completed is not None:
        scan_cmd.extend(["--min-notch-cycles-completed", str(args.scan_min_notch_cycles_completed)])
    if args.max_mechanical_cg_failures is not None:
        scan_cmd.extend(["--max-mechanical-cg-failures", str(args.max_mechanical_cg_failures)])
    if args.max_mechanical_not_accepted_steps is not None:
        scan_cmd.extend(["--max-mechanical-not-accepted-steps", str(args.max_mechanical_not_accepted_steps)])
    if args.max_crack_cg_nonconverged_steps is not None:
        scan_cmd.extend(
            ["--max-crack-cg-nonconverged-steps", str(args.max_crack_cg_nonconverged_steps)]
        )
    tasks.append(("crack_onset_scan", scan_cmd, scan_out / "summary.json"))

    if args.with_exp_alignment:
        exp_out = out_dir / "exp_alignment"
        exp_cmd = [
            py,
            "sim/tests/regress_exp_alignment.py",
            "--config",
            str(args.exp_alignment_config),
            "--out",
            str(exp_out / "summary.json"),
            "--rmse-tau-max",
            str(args.exp_alignment_rmse_tau_max),
            "--mae-tau-max",
            str(args.exp_alignment_mae_tau_max),
            "--rmse-gamma-max",
            str(args.exp_alignment_rmse_gamma_max),
            "--max-runtime-warnings",
            str(args.max_runtime_warnings),
            "--max-mechanical-not-accepted-steps",
            str(args.max_mechanical_not_accepted_steps if args.max_mechanical_not_accepted_steps is not None else 160),
            "--max-crack-cg-nonconverged-steps",
            str(args.max_crack_cg_nonconverged_steps if args.max_crack_cg_nonconverged_steps is not None else 40),
            "--max-nonfinite-count",
            str(args.max_nonfinite_count),
        ]
        tasks.append(("exp_alignment", exp_cmd, exp_out / "summary.json"))

    started_at = datetime.now()
    results: list[TaskResult] = []
    for name, cmd, summary_json in tasks:
        print(f"[run] {name}: {' '.join(cmd)}")
        results.append(_run_task(name, cmd, summary_json))
    finished_at = datetime.now()

    total_runtime_warnings = int(sum(r.runtime_warning_count for r in results))
    passed = all(r.passed for r in results)

    summary = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_s": (finished_at - started_at).total_seconds(),
        "strict": args.strict,
        "max_runtime_warnings": args.max_runtime_warnings,
        "max_mechanical_cg_failures": args.max_mechanical_cg_failures,
        "max_mechanical_not_accepted_steps": args.max_mechanical_not_accepted_steps,
        "max_crack_cg_nonconverged_steps": args.max_crack_cg_nonconverged_steps,
        "max_nonfinite_count": args.max_nonfinite_count,
        "scan_config": str(args.scan_config),
        "scan_min_onset_cases": args.scan_min_onset_cases,
        "scan_min_notch_cycles_completed": args.scan_min_notch_cycles_completed,
        "with_exp_alignment": args.with_exp_alignment,
        "exp_alignment_config": str(args.exp_alignment_config),
        "exp_alignment_rmse_tau_max": args.exp_alignment_rmse_tau_max,
        "exp_alignment_mae_tau_max": args.exp_alignment_mae_tau_max,
        "exp_alignment_rmse_gamma_max": args.exp_alignment_rmse_gamma_max,
        "passed": passed,
        "total_runtime_warning_count": total_runtime_warnings,
        "tasks": [
            {
                "name": r.name,
                "command": r.command,
                "returncode": r.returncode,
                "duration_s": r.duration_s,
                "passed": r.passed,
                "summary_json": r.summary_json,
                "runtime_warning_count": r.runtime_warning_count,
                "failures": r.failures,
            }
            for r in results
        ],
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] phase2 gate summary: {summary_path}")
    print(json.dumps(summary, indent=2))

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
