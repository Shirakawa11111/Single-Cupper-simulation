"""
Run D1 full-gate bundle (non-quick, full-case) and aggregate one summary.

Bundle tasks:
1) phase2_full: phase-2 gate with experiment alignment + energy gate (+ D2 localization by default).
2) multi_align_full: real multi-condition alignment gate (>=3 conditions).
3) d3_multiphysics_matrix: positive/negative matrix gate.
4) seed robustness batches: full-case seed repeat checks.
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

import yaml  # type: ignore


def _default_out_root() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"d1_full_gate_{ts}"


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


def _stringify_seed_batches(batches: list[list[int]]) -> str:
    return ";".join(",".join(str(v) for v in batch) for batch in batches)


def _load_lock(lock_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[list[int]]]:
    raw = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid lock file root: {lock_path}")
    locked_configs = raw.get("locked_configs", {})
    if not isinstance(locked_configs, dict):
        locked_configs = {}
    thresholds = raw.get("thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}
    seed_batches_raw = raw.get("seed_batches", [[41, 42, 43], [44, 45, 46]])
    seed_batches: list[list[int]] = []
    if isinstance(seed_batches_raw, list):
        for row in seed_batches_raw:
            if not isinstance(row, list):
                continue
            vals: list[int] = []
            for v in row:
                vals.append(int(v))
            if vals:
                seed_batches.append(vals)
    if not seed_batches:
        seed_batches = [[41, 42, 43], [44, 45, 46]]
    return locked_configs, thresholds, seed_batches


def _int_val(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float_val(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _path_or(default_path: str, value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return default_path


def _phase2_acceptance(phase2_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(phase2_summary, dict):
        return {"passed": False}
    out: dict[str, Any] = {
        "passed": bool(phase2_summary.get("passed", False)),
        "total_runtime_warning_count": phase2_summary.get("total_runtime_warning_count"),
    }
    tasks = phase2_summary.get("tasks", [])
    if isinstance(tasks, list):
        phase2_task_pass = {}
        for row in tasks:
            if isinstance(row, dict) and isinstance(row.get("name"), str):
                phase2_task_pass[row["name"]] = bool(row.get("passed", False))
        out["task_passed"] = phase2_task_pass
    return out


def _multi_acceptance(multi_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(multi_summary, dict):
        return {"passed": False}
    return {
        "passed": bool(multi_summary.get("passed", False)),
        "condition_total": int(multi_summary.get("condition_total", 0)),
        "passed_count": int(multi_summary.get("passed_count", 0)),
        "failed_names": multi_summary.get("failed_names", []),
    }


def _seed_acceptance(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_summaries:
        return {"enabled": False}
    pass_count = 0
    seed_total = 0
    for row in seed_summaries:
        pass_count += int(row.get("seed_gate_pass_count", 0))
        seed_total += int(row.get("seed_gate_total", 0))
    return {
        "enabled": True,
        "passed": all(bool(row.get("all_seed_gate_passed", False)) for row in seed_summaries),
        "seed_gate_pass_count": pass_count,
        "seed_gate_total": seed_total,
    }


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
    runner_passed = bool(d3_summary.get("passed", False))

    failure_reasons: list[str] = []
    if not runner_passed:
        failure_reasons.append("runner_or_matrix_failed")
    if passed_count < min_pass:
        failure_reasons.append(f"passed_count({passed_count}<{min_pass})")
    if require_all and not (case_total > 0 and passed_count == case_total):
        failure_reasons.append(f"require_all_failed(passed_count={passed_count},case_total={case_total})")

    return {
        "enabled": True,
        "passed": len(failure_reasons) == 0,
        "require_all": bool(require_all),
        "min_pass_count": min_pass,
        "case_total": case_total,
        "passed_count": passed_count,
        "failed_names": failed_names,
        "failure_reasons": failure_reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run D1 full-gate bundle.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--lock-config", type=Path, default=Path("sim/configs/release_pack_v2_lock.yaml"))
    parser.add_argument("--scan-config", type=Path, default=None)
    parser.add_argument("--scan-max-cases", type=int, default=None)
    parser.add_argument("--scan-min-onset-cases", type=int, default=None)
    parser.add_argument("--scan-min-notch-cycles-completed", type=int, default=None)
    parser.add_argument("--exp-alignment-config", type=Path, default=None)
    parser.add_argument("--energy-gate-config", type=Path, default=None)
    parser.add_argument("--d2-localization-config", type=Path, default=None)
    parser.add_argument("--d3-matrix-config", type=Path, default=None)
    parser.add_argument("--seed-base-config", type=Path, default=None)
    parser.add_argument("--multi-config", type=Path, default=None)
    parser.add_argument("--with-phase1-suite", action="store_true")
    parser.add_argument("--skip-d2-localization", action="store_true")
    parser.add_argument("--skip-d3-matrix", action="store_true")
    parser.add_argument("--run-seed-robustness", action="store_true")
    parser.add_argument(
        "--seed-case-mode",
        type=str,
        default="full",
        choices=("pair", "full"),
    )
    parser.add_argument("--seed-batches", type=str, default=None, help='Example: "41,42,43;44,45,46"')
    parser.add_argument("--max-runtime-warnings", type=int, default=None)
    parser.add_argument("--max-mechanical-not-accepted-steps", type=int, default=None)
    parser.add_argument("--max-crack-cg-nonconverged-steps", type=int, default=None)
    parser.add_argument("--max-nonfinite-count", type=int, default=None)
    parser.add_argument("--exp-alignment-rmse-tau-max", type=float, default=None)
    parser.add_argument("--exp-alignment-mae-tau-max", type=float, default=None)
    parser.add_argument("--exp-alignment-rmse-gamma-max", type=float, default=None)
    parser.add_argument("--energy-gate-min-cycles", type=int, default=None)
    parser.add_argument("--d2-min-cycles", type=int, default=None)
    parser.add_argument("--d2-min-crack-delta", type=float, default=None)
    parser.add_argument("--d2-min-localization-index", type=float, default=None)
    parser.add_argument("--d2-min-energy-crack-mean", type=float, default=None)
    parser.add_argument("--d2-min-energy-total-density-mean", type=float, default=None)
    parser.add_argument("--d2-max-runtime-warnings", type=int, default=None)
    parser.add_argument("--d2-max-mechanical-not-accepted-steps", type=int, default=None)
    parser.add_argument("--d2-max-crack-cg-nonconverged-steps", type=int, default=None)
    parser.add_argument("--d2-max-nonfinite-count", type=int, default=None)
    parser.add_argument("--d2-min-vtk-energy-fields", type=int, default=None)
    parser.add_argument("--d3-only", type=str, default=None)
    parser.add_argument("--d3-min-pass-count", type=int, default=None)
    parser.add_argument("--d3-require-all", dest="d3_require_all", action="store_true")
    parser.add_argument("--d3-allow-partial", dest="d3_require_all", action="store_false")
    parser.set_defaults(d3_require_all=None)
    args = parser.parse_args()

    locked_configs, thresholds, lock_seed_batches = _load_lock(args.lock_config)

    scan_config = str(
        args.scan_config
        or _path_or("sim/configs/crack_onset_scan.yaml", locked_configs.get("crack_onset_scan"))
    )
    exp_alignment_config = str(
        args.exp_alignment_config
        or _path_or("sim/configs/fatigue_lowamp_align_locked_v4.yaml", locked_configs.get("exp_alignment"))
    )
    energy_gate_config = str(
        args.energy_gate_config
        or _path_or("sim/configs/fatigue_lowamp_align_locked_v4.yaml", locked_configs.get("energy_gate"))
    )
    d2_localization_config = str(
        args.d2_localization_config
        or _path_or("sim/configs/d2_localization_energy.yaml", locked_configs.get("d2_localization"))
    )
    d3_matrix_config = str(
        args.d3_matrix_config
        or _path_or("sim/configs/d3_multiphysics_matrix.yaml", locked_configs.get("d3_matrix"))
    )
    seed_base_config = str(
        args.seed_base_config
        or _path_or("sim/configs/crack_onset_scan.yaml", locked_configs.get("seed_repeat"))
    )
    multi_config = str(
        args.multi_config
        or _path_or(
            "sim/configs/exp_alignment_multi_d1_full.yaml",
            locked_configs.get("multi_alignment_full") or locked_configs.get("multi_alignment_skeleton"),
        )
    )

    max_runtime_warnings = (
        args.max_runtime_warnings
        if args.max_runtime_warnings is not None
        else _int_val(thresholds.get("max_runtime_warnings"), 50)
    )
    max_mechanical_not_accepted_steps = (
        args.max_mechanical_not_accepted_steps
        if args.max_mechanical_not_accepted_steps is not None
        else _int_val(thresholds.get("max_mechanical_not_accepted_steps"), 160)
    )
    max_crack_cg_nonconverged_steps = (
        args.max_crack_cg_nonconverged_steps
        if args.max_crack_cg_nonconverged_steps is not None
        else _int_val(thresholds.get("max_crack_cg_nonconverged_steps"), 20)
    )
    max_nonfinite_count = (
        args.max_nonfinite_count
        if args.max_nonfinite_count is not None
        else _int_val(thresholds.get("max_nonfinite_count"), 0)
    )
    exp_rmse_tau_max = (
        args.exp_alignment_rmse_tau_max
        if args.exp_alignment_rmse_tau_max is not None
        else _float_val(thresholds.get("exp_alignment_rmse_tau_max"), 30.0)
    )
    exp_mae_tau_max = (
        args.exp_alignment_mae_tau_max
        if args.exp_alignment_mae_tau_max is not None
        else _float_val(thresholds.get("exp_alignment_mae_tau_max"), 25.0)
    )
    exp_rmse_gamma_max = (
        args.exp_alignment_rmse_gamma_max
        if args.exp_alignment_rmse_gamma_max is not None
        else _float_val(thresholds.get("exp_alignment_rmse_gamma_max"), 4.2e-3)
    )
    energy_gate_min_cycles = (
        args.energy_gate_min_cycles
        if args.energy_gate_min_cycles is not None
        else _int_val(thresholds.get("energy_gate_min_cycles"), 5)
    )
    d2_min_cycles = (
        args.d2_min_cycles
        if args.d2_min_cycles is not None
        else _int_val(thresholds.get("d2_min_cycles"), 3)
    )
    d2_min_crack_delta = (
        args.d2_min_crack_delta
        if args.d2_min_crack_delta is not None
        else _float_val(thresholds.get("d2_min_crack_delta"), 5.0e-2)
    )
    d2_min_localization_index = (
        args.d2_min_localization_index
        if args.d2_min_localization_index is not None
        else _float_val(thresholds.get("d2_min_localization_index"), 3.0)
    )
    d2_min_energy_crack_mean = (
        args.d2_min_energy_crack_mean
        if args.d2_min_energy_crack_mean is not None
        else _float_val(thresholds.get("d2_min_energy_crack_mean"), 1.0e-10)
    )
    d2_min_energy_total_density_mean = (
        args.d2_min_energy_total_density_mean
        if args.d2_min_energy_total_density_mean is not None
        else _float_val(thresholds.get("d2_min_energy_total_density_mean"), 1.0e-10)
    )
    d2_max_runtime_warnings = (
        args.d2_max_runtime_warnings
        if args.d2_max_runtime_warnings is not None
        else _int_val(thresholds.get("d2_max_runtime_warnings"), max_runtime_warnings)
    )
    d2_max_mechanical_not_accepted_steps = (
        args.d2_max_mechanical_not_accepted_steps
        if args.d2_max_mechanical_not_accepted_steps is not None
        else _int_val(
            thresholds.get("d2_max_mechanical_not_accepted_steps"),
            max_mechanical_not_accepted_steps,
        )
    )
    d2_max_crack_cg_nonconverged_steps = (
        args.d2_max_crack_cg_nonconverged_steps
        if args.d2_max_crack_cg_nonconverged_steps is not None
        else _int_val(
            thresholds.get("d2_max_crack_cg_nonconverged_steps"),
            max_crack_cg_nonconverged_steps,
        )
    )
    d2_max_nonfinite_count = (
        args.d2_max_nonfinite_count
        if args.d2_max_nonfinite_count is not None
        else _int_val(thresholds.get("d2_max_nonfinite_count"), max_nonfinite_count)
    )
    d2_min_vtk_energy_fields = (
        args.d2_min_vtk_energy_fields
        if args.d2_min_vtk_energy_fields is not None
        else _int_val(thresholds.get("d2_min_vtk_energy_fields"), 4)
    )
    d3_require_all = (
        bool(args.d3_require_all)
        if args.d3_require_all is not None
        else bool(thresholds.get("d3_require_all", True))
    )
    d3_min_pass_count = (
        args.d3_min_pass_count
        if args.d3_min_pass_count is not None
        else _int_val(thresholds.get("d3_min_pass_count"), 3)
    )
    if d3_min_pass_count < 1:
        d3_min_pass_count = 1
    d3_only = (
        args.d3_only
        if args.d3_only is not None
        else str(thresholds.get("d3_only", ""))
    ).strip()

    run_d2_localization = not bool(args.skip_d2_localization)
    run_d3_matrix = not bool(args.skip_d3_matrix)
    run_seed_robustness = bool(args.run_seed_robustness)
    seed_batches = _parse_seed_batches(args.seed_batches) if args.seed_batches else lock_seed_batches

    out_root = args.out_root or _default_out_root()
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now()
    py = args.python
    records: list[dict[str, Any]] = []
    failure_reasons: list[str] = []

    phase2_out = out_root / "phase2_full"
    phase2_cmd = [
        py,
        "sim/tests/regress_phase2.py",
        "--out",
        str(phase2_out),
        "--scan-config",
        scan_config,
        "--max-runtime-warnings",
        str(max_runtime_warnings),
        "--max-mechanical-not-accepted-steps",
        str(max_mechanical_not_accepted_steps),
        "--max-crack-cg-nonconverged-steps",
        str(max_crack_cg_nonconverged_steps),
        "--max-nonfinite-count",
        str(max_nonfinite_count),
        "--with-exp-alignment",
        "--exp-alignment-config",
        exp_alignment_config,
        "--exp-alignment-rmse-tau-max",
        str(exp_rmse_tau_max),
        "--exp-alignment-mae-tau-max",
        str(exp_mae_tau_max),
        "--exp-alignment-rmse-gamma-max",
        str(exp_rmse_gamma_max),
        "--with-energy-gate",
        "--energy-gate-config",
        energy_gate_config,
        "--energy-gate-min-cycles",
        str(energy_gate_min_cycles),
    ]
    if run_d2_localization:
        phase2_cmd.extend(
            [
                "--with-d2-localization",
                "--d2-localization-config",
                d2_localization_config,
                "--d2-min-cycles",
                str(d2_min_cycles),
                "--d2-min-crack-delta",
                str(d2_min_crack_delta),
                "--d2-min-localization-index",
                str(d2_min_localization_index),
                "--d2-min-energy-crack-mean",
                str(d2_min_energy_crack_mean),
                "--d2-min-energy-total-density-mean",
                str(d2_min_energy_total_density_mean),
                "--d2-max-runtime-warnings",
                str(d2_max_runtime_warnings),
                "--d2-max-mechanical-not-accepted-steps",
                str(d2_max_mechanical_not_accepted_steps),
                "--d2-max-crack-cg-nonconverged-steps",
                str(d2_max_crack_cg_nonconverged_steps),
                "--d2-max-nonfinite-count",
                str(d2_max_nonfinite_count),
                "--d2-min-vtk-energy-fields",
                str(d2_min_vtk_energy_fields),
            ]
        )
    if args.scan_max_cases is not None:
        phase2_cmd.extend(["--scan-max-cases", str(args.scan_max_cases)])
    if args.scan_min_onset_cases is not None:
        phase2_cmd.extend(["--scan-min-onset-cases", str(args.scan_min_onset_cases)])
    if args.scan_min_notch_cycles_completed is not None:
        phase2_cmd.extend(["--scan-min-notch-cycles-completed", str(args.scan_min_notch_cycles_completed)])
    if not args.with_phase1_suite:
        phase2_cmd.append("--skip-phase1-suite")
    phase2_rec = _run("phase2_full", phase2_cmd, logs_dir)
    phase2_summary_path = phase2_out / "summary.json"
    phase2_summary = _read_json(phase2_summary_path)
    phase2_passed = bool(
        phase2_rec["runner_passed"] and isinstance(phase2_summary, dict) and phase2_summary.get("passed", False)
    )
    phase2_rec["summary_json"] = str(phase2_summary_path)
    phase2_rec["summary"] = phase2_summary
    phase2_rec["passed"] = phase2_passed
    records.append(phase2_rec)
    if not phase2_passed:
        failure_reasons.append("phase2_full_failed")
        if isinstance(phase2_summary, dict) and isinstance(phase2_summary.get("tasks"), list):
            for row in phase2_summary["tasks"]:
                if isinstance(row, dict) and not bool(row.get("passed", False)):
                    task_name = str(row.get("name", "unknown"))
                    failure_reasons.append(f"phase2_task_failed:{task_name}")
    d2_acceptance = _d2_acceptance(phase2_summary) if run_d2_localization else {"enabled": False, "passed": True}
    if run_d2_localization and not bool(d2_acceptance.get("passed", False)):
        failure_reasons.append("d2_localization_failed")
        d2_failures = d2_acceptance.get("failures")
        if isinstance(d2_failures, dict):
            for key in d2_failures:
                failure_reasons.append(f"d2_failure:{key}")

    d3_acceptance: dict[str, Any]
    if run_d3_matrix:
        d3_out = out_root / "d3_multiphysics_matrix" / "summary.json"
        d3_cmd = [
            py,
            "sim/tests/regress_d3_multiphysics_matrix.py",
            "--config",
            d3_matrix_config,
            "--out",
            str(d3_out),
            "--min-pass-count",
            str(d3_min_pass_count),
            "--require-all" if d3_require_all else "--allow-partial",
        ]
        if d3_only:
            d3_cmd.extend(["--only", d3_only])
        d3_rec = _run("d3_multiphysics_matrix", d3_cmd, logs_dir)
        d3_summary = _read_json(d3_out)
        d3_acceptance = _d3_acceptance(
            d3_summary=d3_summary,
            require_all=d3_require_all,
            min_pass_count=d3_min_pass_count,
        )
        d3_passed = bool(d3_rec["runner_passed"] and bool(d3_acceptance.get("passed", False)))
        d3_rec["summary_json"] = str(d3_out)
        d3_rec["summary"] = d3_summary
        d3_rec["acceptance"] = d3_acceptance
        d3_rec["passed"] = d3_passed
        records.append(d3_rec)
        if not d3_passed:
            failure_reasons.append("d3_multiphysics_matrix_failed")
            d3_failed_names = d3_acceptance.get("failed_names")
            if isinstance(d3_failed_names, list):
                for name in d3_failed_names:
                    failure_reasons.append(f"d3_case_failed:{name}")
            d3_failures = d3_acceptance.get("failure_reasons")
            if isinstance(d3_failures, list):
                for reason in d3_failures:
                    failure_reasons.append(f"d3_failure:{reason}")
    else:
        d3_acceptance = {
            "enabled": False,
            "passed": True,
            "require_all": d3_require_all,
            "min_pass_count": d3_min_pass_count,
        }

    multi_out = out_root / "exp_alignment_multi" / "summary.json"
    multi_cmd = [
        py,
        "sim/tests/regress_exp_alignment_multi.py",
        "--config",
        multi_config,
        "--out",
        str(multi_out),
    ]
    multi_rec = _run("multi_align_full", multi_cmd, logs_dir)
    multi_summary = _read_json(multi_out)
    multi_passed = bool(multi_rec["runner_passed"] and isinstance(multi_summary, dict) and multi_summary.get("passed", False))
    multi_rec["summary_json"] = str(multi_out)
    multi_rec["summary"] = multi_summary
    multi_rec["passed"] = multi_passed
    records.append(multi_rec)
    if not multi_passed:
        failure_reasons.append("multi_align_full_failed")
        if isinstance(multi_summary, dict) and isinstance(multi_summary.get("failed_names"), list):
            for name in multi_summary["failed_names"]:
                failure_reasons.append(f"multi_condition_failed:{name}")

    seed_summaries: list[dict[str, Any]] = []
    if run_seed_robustness:
        for idx, batch in enumerate(seed_batches, start=1):
            batch_name = f"seed_batch_{idx}"
            batch_out = out_root / batch_name
            seeds = ",".join(str(v) for v in batch)
            cmd = [
                py,
                "sim/tests/repeat_crack_onset_seeds.py",
                "--base-config",
                seed_base_config,
                "--case-mode",
                args.seed_case_mode,
                "--seeds",
                seeds,
                "--out-root",
                str(batch_out),
                "--max-runtime-warnings",
                str(max_runtime_warnings),
                "--max-mechanical-not-accepted-steps",
                str(max_mechanical_not_accepted_steps),
                "--max-crack-cg-nonconverged-steps",
                str(max_crack_cg_nonconverged_steps),
                "--max-nonfinite-count",
                str(max_nonfinite_count),
            ]
            rec = _run(batch_name, cmd, logs_dir)
            summary_path = batch_out / "summary.json"
            summary = _read_json(summary_path)
            passed = bool(rec["runner_passed"] and isinstance(summary, dict) and summary.get("all_seed_gate_passed", False))
            rec["summary_json"] = str(summary_path)
            rec["summary"] = summary
            rec["passed"] = passed
            records.append(rec)
            if isinstance(summary, dict):
                seed_summaries.append(summary)
            if not passed:
                failure_reasons.append(f"{batch_name}_failed")

    finished = datetime.now()
    acceptance = {
        "phase2_full": _phase2_acceptance(phase2_summary),
        "d2_localization": d2_acceptance,
        "d3_matrix": d3_acceptance,
        "multi_align_full": _multi_acceptance(multi_summary),
        "seed_robustness": _seed_acceptance(seed_summaries),
    }
    payload = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "duration_s": (finished - started).total_seconds(),
        "out_root": str(out_root),
        "lock_config": str(args.lock_config),
        "configs": {
            "scan_config": scan_config,
            "exp_alignment_config": exp_alignment_config,
            "energy_gate_config": energy_gate_config,
            "d2_localization_config": d2_localization_config,
            "d3_matrix_config": d3_matrix_config,
            "seed_base_config": seed_base_config,
            "multi_config": multi_config,
        },
        "thresholds": {
            "max_runtime_warnings": max_runtime_warnings,
            "max_mechanical_not_accepted_steps": max_mechanical_not_accepted_steps,
            "max_crack_cg_nonconverged_steps": max_crack_cg_nonconverged_steps,
            "max_nonfinite_count": max_nonfinite_count,
            "exp_alignment_rmse_tau_max": exp_rmse_tau_max,
            "exp_alignment_mae_tau_max": exp_mae_tau_max,
            "exp_alignment_rmse_gamma_max": exp_rmse_gamma_max,
            "energy_gate_min_cycles": energy_gate_min_cycles,
            "d2_min_cycles": d2_min_cycles,
            "d2_min_crack_delta": d2_min_crack_delta,
            "d2_min_localization_index": d2_min_localization_index,
            "d2_min_energy_crack_mean": d2_min_energy_crack_mean,
            "d2_min_energy_total_density_mean": d2_min_energy_total_density_mean,
            "d2_max_runtime_warnings": d2_max_runtime_warnings,
            "d2_max_mechanical_not_accepted_steps": d2_max_mechanical_not_accepted_steps,
            "d2_max_crack_cg_nonconverged_steps": d2_max_crack_cg_nonconverged_steps,
            "d2_max_nonfinite_count": d2_max_nonfinite_count,
            "d2_min_vtk_energy_fields": d2_min_vtk_energy_fields,
            "d3_require_all": d3_require_all,
            "d3_min_pass_count": d3_min_pass_count,
        },
        "run_d2_localization": run_d2_localization,
        "run_d3_matrix": run_d3_matrix,
        "run_seed_robustness": run_seed_robustness,
        "d3_only": d3_only if run_d3_matrix and d3_only else None,
        "seed_case_mode": args.seed_case_mode if run_seed_robustness else None,
        "seed_batches": _stringify_seed_batches(seed_batches) if run_seed_robustness else None,
        "acceptance": acceptance,
        "tasks": records,
        "failure_reasons": failure_reasons,
        "passed": len(records) > 0 and all(bool(r.get("passed", False)) for r in records),
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary_path)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
