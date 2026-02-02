"""
Unified regression entry for boundary handling and crack driving.

Runs small, large, and micron regression scripts and aggregates JSON reports.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TESTS = {
    "small": Path(__file__).with_name("regress_bc_crack.py"),
    "large": Path(__file__).with_name("regress_bc_crack_large.py"),
    "micron": Path(__file__).with_name("regress_bc_crack_micron.py"),
}


def _run_one(name: str, script: Path, strict: bool, log_dir: Path | None) -> Dict[str, Any]:
    cmd = [sys.executable, str(script)]
    if strict:
        cmd.append("--strict")

    output_path = None
    if log_dir is not None:
        output_path = log_dir / f"{name}.json"
        cmd.extend(["--output", str(output_path)])

    t0 = perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_s = perf_counter() - t0

    report = None
    if output_path is not None and output_path.exists():
        try:
            report = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = None

    if report is None:
        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError:
            report = {
                "passed": False,
                "failures": {"runner": "no JSON report"},
            }

    if log_dir is not None:
        (log_dir / f"{name}.stdout").write_text(proc.stdout, encoding="utf-8")
        (log_dir / f"{name}.stderr").write_text(proc.stderr, encoding="utf-8")

    passed = bool(report.get("passed", False)) and proc.returncode == 0

    return {
        "name": name,
        "script": str(script),
        "passed": passed,
        "returncode": proc.returncode,
        "timing": {"wall_s": wall_s},
        "report": report,
        "stdout_tail": proc.stdout.strip()[-2000:],
        "stderr_tail": proc.stderr.strip()[-2000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="", help="Optional JSON summary output path")
    parser.add_argument("--strict", action="store_true", help="Use stricter thresholds for each test")
    parser.add_argument("--small", action="store_true", help="Run small regression only")
    parser.add_argument("--large", action="store_true", help="Run large regression only")
    parser.add_argument("--micron", action="store_true", help="Run micron regression only")
    parser.add_argument(
        "--log-dir",
        default="",
        help="Optional directory to store per-test JSON/stdout/stderr logs",
    )
    args = parser.parse_args()

    selection = []
    if args.small or args.large or args.micron:
        if args.small:
            selection.append("small")
        if args.large:
            selection.append("large")
        if args.micron:
            selection.append("micron")
    else:
        selection = ["small", "large", "micron"]

    log_dir = None
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    failures: Dict[str, Any] = {}
    overall_passed = True

    total_start = perf_counter()
    for name in selection:
        script = TESTS[name]
        result = _run_one(name, script, args.strict, log_dir)
        results[name] = result
        if not result["passed"]:
            overall_passed = False
            failures[name] = result.get("report", {}).get("failures", {})
    total_s = perf_counter() - total_start

    payload = {
        "passed": overall_passed,
        "tests": results,
        "failures": failures,
        "timing": {"total_s": total_s},
        "strict": args.strict,
        "selection": selection,
        "log_dir": str(log_dir) if log_dir is not None else "",
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
