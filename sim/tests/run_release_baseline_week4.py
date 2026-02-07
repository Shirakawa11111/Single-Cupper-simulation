"""
Run Week-4 baseline release bundle.

Bundle includes:
1) phase2 + exp-alignment gate (quick/full profiles)
2) optional seed-robustness batches
3) consolidated bundle summary.json
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
    return Path("sim/tests/regress_runs") / day / f"release_baseline_week4_{ts}"


def _parse_seed_batches(text: str) -> list[list[int]]:
    batches: list[list[int]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals: list[int] = []
        for part in chunk.split(","):
            part = part.strip()
            if not part:
                continue
            vals.append(int(part))
        if vals:
            batches.append(vals)
    if not batches:
        raise ValueError("seed batch string is empty")
    return batches


def _run(name: str, cmd: list[str], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = perf_counter() - t0
    (out_dir / f"{name}.stdout").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"{name}.stderr").write_text(proc.stderr, encoding="utf-8")
    return {
        "name": name,
        "command": cmd,
        "returncode": int(proc.returncode),
        "duration_s": dt,
        "stdout_log": str(out_dir / f"{name}.stdout"),
        "stderr_log": str(out_dir / f"{name}.stderr"),
        "passed": proc.returncode == 0,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run week4 release baseline bundle.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument(
        "--profile",
        type=str,
        default="quick",
        choices=("quick", "full_skip_phase1", "full"),
        help="Gate profile: quick/ full_skip_phase1 / full.",
    )
    parser.add_argument("--run-seed-robustness", action="store_true")
    parser.add_argument(
        "--seed-case-mode",
        type=str,
        default="pair",
        choices=("pair", "full"),
        help="Seed robustness mode: pair uses notch+negative pair, full uses all configured cases.",
    )
    parser.add_argument(
        "--seed-batches",
        type=str,
        default="41,42,43;44,45,46",
        help="Semicolon-separated seed batches, each batch is comma-separated ints.",
    )
    parser.add_argument("--max-runtime-warnings", type=int, default=50)
    parser.add_argument("--max-mechanical-not-accepted-steps", type=int, default=160)
    parser.add_argument("--max-crack-cg-nonconverged-steps", type=int, default=20)
    parser.add_argument("--max-nonfinite-count", type=int, default=0)
    args = parser.parse_args()

    py = args.python
    out_root = args.out_root or _default_out_root()
    out_root.mkdir(parents=True, exist_ok=True)
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    started = datetime.now()
    records: list[dict[str, Any]] = []

    phase2_out = out_root / "phase2_gate"
    phase2_cmd = [
        py,
        "sim/tests/regress_phase2.py",
        "--out",
        str(phase2_out),
        "--with-exp-alignment",
        "--exp-alignment-config",
        "sim/configs/fatigue_lowamp_align_locked_v4.yaml",
        "--max-runtime-warnings",
        str(args.max_runtime_warnings),
        "--max-nonfinite-count",
        str(args.max_nonfinite_count),
    ]
    if args.profile == "quick":
        phase2_cmd.extend(
            [
                "--skip-phase1-suite",
                "--scan-config",
                "sim/configs/crack_onset_scan_quick.yaml",
                "--scan-max-cases",
                "1",
                "--scan-min-onset-cases",
                "1",
                "--max-crack-cg-nonconverged-steps",
                "40",
            ]
        )
    elif args.profile == "full_skip_phase1":
        phase2_cmd.extend(
            [
                "--skip-phase1-suite",
                "--scan-config",
                "sim/configs/crack_onset_scan.yaml",
                "--max-mechanical-not-accepted-steps",
                str(args.max_mechanical_not_accepted_steps),
                "--max-crack-cg-nonconverged-steps",
                str(args.max_crack_cg_nonconverged_steps),
            ]
        )
    else:
        phase2_cmd.extend(
            [
                "--strict",
                "--scan-config",
                "sim/configs/crack_onset_scan.yaml",
                "--max-mechanical-not-accepted-steps",
                str(args.max_mechanical_not_accepted_steps),
                "--max-crack-cg-nonconverged-steps",
                str(args.max_crack_cg_nonconverged_steps),
            ]
        )
    rec = _run("phase2_gate", phase2_cmd, logs)
    rec["summary_json"] = str(phase2_out / "summary.json")
    rec["summary"] = _read_json(phase2_out / "summary.json")
    rec["passed"] = bool(rec["passed"] and isinstance(rec["summary"], dict) and rec["summary"].get("passed", False))
    records.append(rec)

    if args.run_seed_robustness:
        batches = _parse_seed_batches(args.seed_batches)
        for idx, batch in enumerate(batches, start=1):
            tag = f"seed_batch_{idx}"
            batch_out = out_root / tag
            seeds = ",".join(str(v) for v in batch)
            cmd = [
                py,
                "sim/tests/repeat_crack_onset_seeds.py",
                "--base-config",
                "sim/configs/crack_onset_scan.yaml",
                "--seeds",
                seeds,
                "--case-mode",
                args.seed_case_mode,
                "--notch-case",
                "control_notch_mild",
                "--negative-case",
                "no_notch_control",
                "--out-root",
                str(batch_out),
                "--max-runtime-warnings",
                str(args.max_runtime_warnings),
                "--max-mechanical-not-accepted-steps",
                str(args.max_mechanical_not_accepted_steps),
                "--max-crack-cg-nonconverged-steps",
                str(args.max_crack_cg_nonconverged_steps),
                "--max-nonfinite-count",
                str(args.max_nonfinite_count),
            ]
            rec = _run(tag, cmd, logs)
            rec["summary_json"] = str(batch_out / "summary.json")
            rec["summary"] = _read_json(batch_out / "summary.json")
            rec["passed"] = bool(
                rec["passed"]
                and isinstance(rec["summary"], dict)
                and rec["summary"].get("all_seed_gate_passed", False)
            )
            records.append(rec)

    finished = datetime.now()
    bundle = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "duration_s": (finished - started).total_seconds(),
        "profile": args.profile,
        "run_seed_robustness": args.run_seed_robustness,
        "seed_case_mode": args.seed_case_mode if args.run_seed_robustness else None,
        "seed_batches": args.seed_batches if args.run_seed_robustness else None,
        "out_root": str(out_root),
        "tasks": records,
        "passed": all(bool(r.get("passed", False)) for r in records),
    }
    summary_path = out_root / "bundle_summary.json"
    summary_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"[done] {summary_path}")
    print(json.dumps(bundle, indent=2))
    return 0 if bundle["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
