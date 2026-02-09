"""
Run Week-4 baseline release bundle.

Bundle includes:
1) phase2 + exp-alignment gate (quick/full profiles, with D2 localization by default)
2) D3 multi-physics matrix gate (enabled by default)
3) optional seed-robustness batches
4) consolidated bundle summary.json
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


def _d2_acceptance(phase2_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(phase2_summary, dict):
        return {"enabled": False, "passed": False}
    enabled = bool(phase2_summary.get("with_d2_localization", False))
    tasks = phase2_summary.get("tasks")
    if not isinstance(tasks, list):
        return {"enabled": enabled, "passed": False}
    d2_task = None
    for row in tasks:
        if isinstance(row, dict) and str(row.get("name")) == "d2_localization":
            d2_task = row
            break
    if not isinstance(d2_task, dict):
        return {"enabled": enabled, "passed": False}
    out: dict[str, Any] = {
        "enabled": enabled,
        "passed": bool(d2_task.get("passed", False)),
        "summary_json": d2_task.get("summary_json"),
    }
    summary_path = d2_task.get("summary_json")
    if isinstance(summary_path, str) and summary_path.strip():
        d2_summary = _read_json(Path(summary_path))
        if isinstance(d2_summary, dict):
            failures = d2_summary.get("failures")
            if isinstance(failures, dict):
                out["failures"] = failures
            metrics = d2_summary.get("metrics")
            if isinstance(metrics, dict):
                for key in (
                    "cycles_completed",
                    "crack_delta_total",
                    "crack_localization_index_peak",
                    "energy_crack_mean_final",
                    "energy_total_density_mean_final",
                    "vtk_energy_field_count",
                ):
                    if key in metrics:
                        out[key] = metrics.get(key)
    return out


def _d3_acceptance(d3_summary: dict[str, Any] | None, require_all: bool, min_pass_count: int) -> dict[str, Any]:
    min_pass = max(1, int(min_pass_count))
    if not isinstance(d3_summary, dict):
        return {
            "enabled": True,
            "passed": False,
            "require_all": bool(require_all),
            "min_pass_count": min_pass,
            "failure_reasons": ["summary_missing_or_invalid"],
        }
    case_total = int(d3_summary.get("case_total", 0))
    passed_count = int(d3_summary.get("passed_count", 0))
    failed_names_raw = d3_summary.get("failed_names")
    failed_names = failed_names_raw if isinstance(failed_names_raw, list) else []
    matrix_passed = bool(d3_summary.get("passed", False))

    failures: list[str] = []
    if not matrix_passed:
        failures.append("runner_or_matrix_failed")
    if passed_count < min_pass:
        failures.append(f"passed_count({passed_count}<{min_pass})")
    if require_all and not (case_total > 0 and passed_count == case_total):
        failures.append(f"require_all_failed(passed_count={passed_count},case_total={case_total})")

    return {
        "enabled": True,
        "passed": len(failures) == 0,
        "require_all": bool(require_all),
        "min_pass_count": min_pass,
        "case_total": case_total,
        "passed_count": passed_count,
        "failed_names": failed_names,
        "failure_reasons": failures,
    }


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
    parser.add_argument("--skip-d2-localization", action="store_true")
    parser.add_argument(
        "--d2-localization-config",
        type=Path,
        default=Path("sim/configs/d2_localization_energy.yaml"),
    )
    parser.add_argument("--d2-min-cycles", type=int, default=3)
    parser.add_argument("--d2-min-crack-delta", type=float, default=5.0e-2)
    parser.add_argument("--d2-min-localization-index", type=float, default=3.0)
    parser.add_argument("--d2-min-energy-crack-mean", type=float, default=1.0e-10)
    parser.add_argument("--d2-min-energy-total-density-mean", type=float, default=1.0e-10)
    parser.add_argument("--d2-max-runtime-warnings", type=int, default=None)
    parser.add_argument("--d2-max-mechanical-not-accepted-steps", type=int, default=None)
    parser.add_argument("--d2-max-crack-cg-nonconverged-steps", type=int, default=None)
    parser.add_argument("--d2-max-nonfinite-count", type=int, default=None)
    parser.add_argument("--d2-min-vtk-energy-fields", type=int, default=4)
    parser.add_argument("--skip-d3-matrix", action="store_true")
    parser.add_argument("--d3-matrix-config", type=Path, default=Path("sim/configs/d3_multiphysics_matrix.yaml"))
    parser.add_argument("--d3-only", type=str, default="")
    parser.add_argument("--d3-min-pass-count", type=int, default=3)
    parser.add_argument("--d3-require-all", dest="d3_require_all", action="store_true")
    parser.add_argument("--d3-allow-partial", dest="d3_require_all", action="store_false")
    parser.set_defaults(d3_require_all=True)
    args = parser.parse_args()

    py = args.python
    run_d2_localization = not bool(args.skip_d2_localization)
    run_d3_matrix = not bool(args.skip_d3_matrix)
    d3_min_pass_count = max(1, int(args.d3_min_pass_count))
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
    if run_d2_localization:
        phase2_cmd.extend(
            [
                "--with-d2-localization",
                "--d2-localization-config",
                str(args.d2_localization_config),
                "--d2-min-cycles",
                str(args.d2_min_cycles),
                "--d2-min-crack-delta",
                str(args.d2_min_crack_delta),
                "--d2-min-localization-index",
                str(args.d2_min_localization_index),
                "--d2-min-energy-crack-mean",
                str(args.d2_min_energy_crack_mean),
                "--d2-min-energy-total-density-mean",
                str(args.d2_min_energy_total_density_mean),
                "--d2-max-runtime-warnings",
                str(args.d2_max_runtime_warnings if args.d2_max_runtime_warnings is not None else args.max_runtime_warnings),
                "--d2-max-mechanical-not-accepted-steps",
                str(
                    args.d2_max_mechanical_not_accepted_steps
                    if args.d2_max_mechanical_not_accepted_steps is not None
                    else args.max_mechanical_not_accepted_steps
                ),
                "--d2-max-crack-cg-nonconverged-steps",
                str(
                    args.d2_max_crack_cg_nonconverged_steps
                    if args.d2_max_crack_cg_nonconverged_steps is not None
                    else args.max_crack_cg_nonconverged_steps
                ),
                "--d2-max-nonfinite-count",
                str(args.d2_max_nonfinite_count if args.d2_max_nonfinite_count is not None else args.max_nonfinite_count),
                "--d2-min-vtk-energy-fields",
                str(args.d2_min_vtk_energy_fields),
            ]
        )
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
    phase2_summary = rec["summary"] if isinstance(rec.get("summary"), dict) else None
    d2_acceptance = _d2_acceptance(phase2_summary) if run_d2_localization else {"enabled": False, "passed": True}

    if run_d3_matrix:
        d3_out = out_root / "d3_multiphysics_matrix" / "summary.json"
        d3_cmd = [
            py,
            "sim/tests/regress_d3_multiphysics_matrix.py",
            "--config",
            str(args.d3_matrix_config),
            "--out",
            str(d3_out),
            "--min-pass-count",
            str(d3_min_pass_count),
            "--require-all" if args.d3_require_all else "--allow-partial",
        ]
        if args.d3_only.strip():
            d3_cmd.extend(["--only", args.d3_only.strip()])
        d3_rec = _run("d3_multiphysics_matrix", d3_cmd, logs)
        d3_rec["summary_json"] = str(d3_out)
        d3_rec["summary"] = _read_json(d3_out)
        d3_acceptance = _d3_acceptance(
            d3_summary=d3_rec["summary"] if isinstance(d3_rec["summary"], dict) else None,
            require_all=bool(args.d3_require_all),
            min_pass_count=d3_min_pass_count,
        )
        d3_rec["acceptance"] = d3_acceptance
        d3_rec["passed"] = bool(d3_rec["passed"] and d3_acceptance.get("passed", False))
        records.append(d3_rec)
    else:
        d3_acceptance = {
            "enabled": False,
            "passed": True,
            "require_all": bool(args.d3_require_all),
            "min_pass_count": d3_min_pass_count,
        }

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
        "run_d2_localization": run_d2_localization,
        "run_d3_matrix": run_d3_matrix,
        "d2_localization_config": str(args.d2_localization_config) if run_d2_localization else None,
        "d3_matrix_config": str(args.d3_matrix_config) if run_d3_matrix else None,
        "d2_thresholds": (
            {
                "min_cycles": args.d2_min_cycles,
                "min_crack_delta": args.d2_min_crack_delta,
                "min_localization_index": args.d2_min_localization_index,
                "min_energy_crack_mean": args.d2_min_energy_crack_mean,
                "min_energy_total_density_mean": args.d2_min_energy_total_density_mean,
                "max_runtime_warnings": (
                    args.d2_max_runtime_warnings if args.d2_max_runtime_warnings is not None else args.max_runtime_warnings
                ),
                "max_mechanical_not_accepted_steps": (
                    args.d2_max_mechanical_not_accepted_steps
                    if args.d2_max_mechanical_not_accepted_steps is not None
                    else args.max_mechanical_not_accepted_steps
                ),
                "max_crack_cg_nonconverged_steps": (
                    args.d2_max_crack_cg_nonconverged_steps
                    if args.d2_max_crack_cg_nonconverged_steps is not None
                    else args.max_crack_cg_nonconverged_steps
                ),
                "max_nonfinite_count": (
                    args.d2_max_nonfinite_count if args.d2_max_nonfinite_count is not None else args.max_nonfinite_count
                ),
                "min_vtk_energy_fields": args.d2_min_vtk_energy_fields,
            }
            if run_d2_localization
            else None
        ),
        "d3_thresholds": (
            {
                "require_all": bool(args.d3_require_all),
                "min_pass_count": d3_min_pass_count,
                "only": args.d3_only.strip() if args.d3_only.strip() else None,
            }
            if run_d3_matrix
            else None
        ),
        "run_seed_robustness": args.run_seed_robustness,
        "seed_case_mode": args.seed_case_mode if args.run_seed_robustness else None,
        "seed_batches": args.seed_batches if args.run_seed_robustness else None,
        "acceptance": {
            "d2_localization": d2_acceptance,
            "d3_matrix": d3_acceptance,
        },
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
