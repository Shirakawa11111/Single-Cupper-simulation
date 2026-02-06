"""
Run the Phase-1 regression suite and write an aggregated summary JSON.
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
    return Path("sim/tests/regress_runs") / day / f"phase1_suite_{ts}"


@dataclass
class TaskResult:
    name: str
    command: list[str]
    returncode: int
    duration_s: float
    passed: bool
    output_json: str
    runtime_warning_count: int
    runtime_warning_items: list[dict[str, Any]]
    failures: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _run_task(name: str, cmd: list[str], output_json: Path) -> TaskResult:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dur = perf_counter() - t0

    stdout_path = output_json.with_suffix(".stdout")
    stderr_path = output_json.with_suffix(".stderr")
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    report = _read_json(output_json)
    passed = proc.returncode == 0 and bool(report is not None and report.get("passed", True))
    failures = {}
    runtime_warning_count = proc.stdout.count("RuntimeWarning") + proc.stderr.count("RuntimeWarning")
    runtime_warning_items: list[dict[str, Any]] = []
    if report is not None:
        failures = report.get("failures", {}) if isinstance(report.get("failures", {}), dict) else {}
        report_warn = report.get("runtime_warning_count")
        if isinstance(report_warn, int):
            runtime_warning_count = max(runtime_warning_count, int(report_warn))
        raw_items = report.get("runtime_warning_items")
        if isinstance(raw_items, list):
            runtime_warning_items = [
                {"message": str(item.get("message", "")), "count": int(item.get("count", 0))}
                for item in raw_items
                if isinstance(item, dict)
            ]
    if proc.returncode != 0 and not failures:
        failures = {"runner": f"returncode={proc.returncode}"}

    return TaskResult(
        name=name,
        command=cmd,
        returncode=proc.returncode,
        duration_s=dur,
        passed=passed,
        output_json=str(output_json),
        runtime_warning_count=runtime_warning_count,
        runtime_warning_items=runtime_warning_items,
        failures=failures,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase-1 regression suite.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for logs and summary.")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode for boundary_crack regression.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable path.")
    parser.add_argument(
        "--max-runtime-warnings",
        type=int,
        default=None,
        help="Fail suite when total RuntimeWarning count exceeds this threshold.",
    )
    args = parser.parse_args()

    out_dir = args.out or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    py = args.python
    tasks: list[tuple[str, list[str], Path]] = [
        (
            "microstrain",
            [py, "sim/tests/regress_microstrain.py", "--output", str(out_dir / "microstrain" / "summary.json")],
            out_dir / "microstrain" / "summary.json",
        ),
        (
            "gnd",
            [py, "sim/tests/regress_gnd.py", "--output", str(out_dir / "gnd" / "summary.json")],
            out_dir / "gnd" / "summary.json",
        ),
        (
            "gnd_cycle",
            [py, "sim/tests/regress_gnd_cycle.py", "--output", str(out_dir / "gnd_cycle" / "summary.json")],
            out_dir / "gnd_cycle" / "summary.json",
        ),
    ]

    bc_dir = out_dir / "boundary_crack"
    bc_cmd = [
        py,
        "sim/tests/regress_all.py",
        "--log-dir",
        str(bc_dir),
        "--output",
        str(bc_dir / "summary.json"),
    ]
    if args.strict:
        bc_cmd.append("--strict")
    tasks.append(("boundary_crack", bc_cmd, bc_dir / "summary.json"))

    started_at = datetime.now()
    task_results: list[TaskResult] = []
    for name, cmd, out_json in tasks:
        print(f"[run] {name}: {' '.join(cmd)}")
        task_results.append(_run_task(name, cmd, out_json))

    finished_at = datetime.now()
    passed = all(t.passed for t in task_results)
    total_runtime_warnings = int(sum(t.runtime_warning_count for t in task_results))
    failure_reasons: list[str] = []
    if args.max_runtime_warnings is not None and total_runtime_warnings > args.max_runtime_warnings:
        passed = False
        failure_reasons.append(
            f"runtime_warning_count_exceeded({total_runtime_warnings}>{args.max_runtime_warnings})"
        )

    summary = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_s": (finished_at - started_at).total_seconds(),
        "strict": args.strict,
        "max_runtime_warnings": args.max_runtime_warnings,
        "total_runtime_warning_count": total_runtime_warnings,
        "passed": passed,
        "failure_reasons": failure_reasons,
        "tasks": [
            {
                "name": t.name,
                "command": t.command,
                "returncode": t.returncode,
                "duration_s": t.duration_s,
                "passed": t.passed,
                "output_json": t.output_json,
                "runtime_warning_count": t.runtime_warning_count,
                "runtime_warning_items": t.runtime_warning_items,
                "failures": t.failures,
            }
            for t in task_results
        ],
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] phase1 suite summary: {summary_path}")
    print(json.dumps(summary, indent=2))

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
