"""
D3 multi-physics matrix gate.

Runs a small matrix of D2 localization-energy cases (positive/negative controls),
then evaluates case-level expectations and emits one aggregated summary.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml  # type: ignore


def _default_out() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"d3_multiphysics_matrix_{ts}" / "summary.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)
        else:
            out[key] = val
    return out


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _runner_cmd(
    py: str,
    runner_script: str,
    config_path: Path,
    out_path: Path,
    runner_args: dict[str, Any],
) -> list[str]:
    cmd = [py, runner_script, "--config", str(config_path), "--out", str(out_path)]
    flags = {
        "min_cycles": "--min-cycles",
        "min_crack_delta": "--min-crack-delta",
        "min_localization_index": "--min-localization-index",
        "min_energy_crack_mean": "--min-energy-crack-mean",
        "min_energy_total_density_mean": "--min-energy-total-density-mean",
        "max_runtime_warnings": "--max-runtime-warnings",
        "max_mechanical_not_accepted_steps": "--max-mechanical-not-accepted-steps",
        "max_crack_cg_nonconverged_steps": "--max-crack-cg-nonconverged-steps",
        "max_nonfinite_count": "--max-nonfinite-count",
        "min_vtk_energy_fields": "--min-vtk-energy-fields",
    }
    for key, flag in flags.items():
        if key not in runner_args:
            continue
        value = runner_args.get(key)
        if value is None:
            continue
        cmd.extend([flag, str(value)])
    return cmd


def _eval_positive(report: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str], dict[str, float]]:
    failures: list[str] = []
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    crack_delta = _as_float(metrics.get("crack_delta_total"))
    crack_mean_final = _as_float(metrics.get("crack_mean_final"))
    crack_loc_peak = _as_float(metrics.get("crack_localization_index_peak"))
    energy_total = _as_float(metrics.get("energy_total_density_mean_final"))
    energy_crack = _as_float(metrics.get("energy_crack_mean_final"))
    vtk_count = _as_float(metrics.get("vtk_energy_field_count"))

    min_crack_delta = _as_float(expected.get("min_crack_delta"), 5.0e-2)
    min_crack_mean_final = _as_float(expected.get("min_crack_mean_final"), float("nan"))
    min_localization = _as_float(expected.get("min_localization_index"), 3.0)
    min_energy_total = _as_float(expected.get("min_energy_total_density_mean"), 1.0e-10)
    min_energy_crack = _as_float(expected.get("min_energy_crack_mean"), 1.0e-10)
    min_vtk = _as_float(expected.get("min_vtk_energy_fields"), 4.0)

    if not _is_finite(crack_delta) or crack_delta < min_crack_delta:
        failures.append(f"crack_delta_total({crack_delta:.6e}<{min_crack_delta:.6e})")
    if _is_finite(min_crack_mean_final):
        if not _is_finite(crack_mean_final) or crack_mean_final < min_crack_mean_final:
            failures.append(f"crack_mean_final({crack_mean_final:.6e}<{min_crack_mean_final:.6e})")
    if not _is_finite(crack_loc_peak) or crack_loc_peak < min_localization:
        failures.append(f"crack_localization_index_peak({crack_loc_peak:.6e}<{min_localization:.6e})")
    if not _is_finite(energy_total) or energy_total < min_energy_total:
        failures.append(f"energy_total_density_mean_final({energy_total:.6e}<{min_energy_total:.6e})")
    if not _is_finite(energy_crack) or energy_crack < min_energy_crack:
        failures.append(f"energy_crack_mean_final({energy_crack:.6e}<{min_energy_crack:.6e})")
    if not _is_finite(vtk_count) or vtk_count < min_vtk:
        failures.append(f"vtk_energy_field_count({vtk_count:.0f}<{min_vtk:.0f})")

    snap = {
        "crack_delta_total": crack_delta,
        "crack_mean_final": crack_mean_final,
        "crack_localization_index_peak": crack_loc_peak,
        "energy_total_density_mean_final": energy_total,
        "energy_crack_mean_final": energy_crack,
        "vtk_energy_field_count": vtk_count,
    }
    return len(failures) == 0, failures, snap


def _eval_negative(report: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str], dict[str, float]]:
    failures: list[str] = []
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    crack_delta = _as_float(metrics.get("crack_delta_total"))
    crack_mean_final = _as_float(metrics.get("crack_mean_final"))
    crack_loc_peak = _as_float(metrics.get("crack_localization_index_peak"))
    energy_total = _as_float(metrics.get("energy_total_density_mean_final"))

    max_crack_delta = _as_float(expected.get("max_crack_delta"), 5.0e-2)
    max_crack_mean_final = _as_float(expected.get("max_crack_mean_final"), 8.0e-2)
    max_localization = _as_float(expected.get("max_localization_index"), float("nan"))
    max_energy_total = _as_float(expected.get("max_energy_total_density_mean"), float("nan"))

    if not _is_finite(crack_delta) or crack_delta > max_crack_delta:
        failures.append(f"crack_delta_total({crack_delta:.6e}>{max_crack_delta:.6e})")
    if not _is_finite(crack_mean_final) or crack_mean_final > max_crack_mean_final:
        failures.append(f"crack_mean_final({crack_mean_final:.6e}>{max_crack_mean_final:.6e})")
    if _is_finite(max_localization):
        if not _is_finite(crack_loc_peak) or crack_loc_peak > max_localization:
            failures.append(f"crack_localization_index_peak({crack_loc_peak:.6e}>{max_localization:.6e})")
    if _is_finite(max_energy_total):
        if not _is_finite(energy_total) or energy_total > max_energy_total:
            failures.append(f"energy_total_density_mean_final({energy_total:.6e}>{max_energy_total:.6e})")

    snap = {
        "crack_delta_total": crack_delta,
        "crack_mean_final": crack_mean_final,
        "crack_localization_index_peak": crack_loc_peak,
        "energy_total_density_mean_final": energy_total,
    }
    return len(failures) == 0, failures, snap


def main() -> int:
    parser = argparse.ArgumentParser(description="Run D3 multi-physics matrix gate.")
    parser.add_argument("--config", type=Path, default=Path("sim/configs/d3_multiphysics_matrix.yaml"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--only", type=str, default="", help="Optional comma-separated case names.")
    parser.add_argument(
        "--require-all",
        dest="require_all_override",
        action="store_true",
        help="Override config and require all enabled cases passing.",
    )
    parser.add_argument(
        "--allow-partial",
        dest="require_all_override",
        action="store_false",
        help="Override config and allow partial pass by min-pass-count.",
    )
    parser.add_argument("--min-pass-count", type=int, default=None, help="Override minimum passed case count.")
    parser.set_defaults(require_all_override=None)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")

    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    default_base_config = Path(defaults.get("base_config", "sim/configs/d2_localization_energy.yaml"))
    default_runner_script = str(defaults.get("runner_script", "sim/tests/regress_d2_localization_energy.py"))
    default_runner_args = defaults.get("runner_args", {})
    if not isinstance(default_runner_args, dict):
        default_runner_args = {}
    default_root_overrides = defaults.get("config_overrides", {})
    if not isinstance(default_root_overrides, dict):
        default_root_overrides = {}
    default_vc_overrides = defaults.get("virtual_cycle_overrides", {})
    if not isinstance(default_vc_overrides, dict):
        default_vc_overrides = {}
    expectations = defaults.get("expectations", {})
    if not isinstance(expectations, dict):
        expectations = {}
    expected_positive = expectations.get("positive", {})
    if not isinstance(expected_positive, dict):
        expected_positive = {}
    expected_negative = expectations.get("negative", {})
    if not isinstance(expected_negative, dict):
        expected_negative = {}
    config_require_all = bool(defaults.get("require_all", True))
    config_min_pass_count = max(1, _as_int(defaults.get("min_pass_count"), 1))
    require_all = config_require_all if args.require_all_override is None else bool(args.require_all_override)
    min_pass_count = max(
        1,
        _as_int(args.min_pass_count, config_min_pass_count)
        if args.min_pass_count is not None
        else config_min_pass_count,
    )

    cases = raw.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list.")

    selected = {name.strip() for name in args.only.split(",") if name.strip()}
    out_path = args.out or _default_out()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    case_root = out_path.parent / "cases"
    log_root = out_path.parent / "logs"
    case_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    started = datetime.now()
    records: list[dict[str, Any]] = []

    for idx, case_raw in enumerate(cases, start=1):
        if not isinstance(case_raw, dict):
            continue
        name = str(case_raw.get("name", f"case_{idx:02d}")).strip()
        if not name:
            continue
        if selected and name not in selected:
            continue
        if not bool(case_raw.get("enabled", True)):
            continue

        mode = str(case_raw.get("mode", "positive")).strip().lower()
        if mode not in ("positive", "negative"):
            raise ValueError(f"case `{name}` mode must be positive/negative.")

        base_config = Path(case_raw.get("base_config", default_base_config))
        cfg_raw = yaml.safe_load(base_config.read_text(encoding="utf-8"))
        if not isinstance(cfg_raw, dict):
            raise ValueError(f"base_config `{base_config}` root must be a mapping.")

        root_overrides = dict(default_root_overrides)
        case_root_overrides = case_raw.get("config_overrides", {})
        if isinstance(case_root_overrides, dict):
            root_overrides = _merge(root_overrides, case_root_overrides)
        vc_overrides = dict(default_vc_overrides)
        case_vc_overrides = case_raw.get("virtual_cycle_overrides", {})
        if isinstance(case_vc_overrides, dict):
            vc_overrides = _merge(vc_overrides, case_vc_overrides)

        resolved_cfg = _merge(cfg_raw, root_overrides)
        if "virtual_cycle" in resolved_cfg:
            vc = resolved_cfg.get("virtual_cycle")
            if not isinstance(vc, dict):
                raise ValueError(f"base_config `{base_config}` virtual_cycle must be mapping.")
            resolved_cfg["virtual_cycle"] = _merge(vc, vc_overrides)
        else:
            resolved_cfg = _merge(resolved_cfg, vc_overrides)

        case_dir = case_root / f"{idx:02d}_{name}"
        case_dir.mkdir(parents=True, exist_ok=True)
        resolved_cfg_path = case_dir / "resolved_config.yaml"
        resolved_cfg_path.write_text(yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8")

        runner_args = dict(default_runner_args)
        case_runner_args = case_raw.get("runner_args", {})
        if isinstance(case_runner_args, dict):
            runner_args = _merge(runner_args, case_runner_args)

        summary_path = case_dir / "summary.json"
        runner_script = str(case_raw.get("runner_script", default_runner_script))
        cmd = _runner_cmd(
            py=args.python,
            runner_script=runner_script,
            config_path=resolved_cfg_path,
            out_path=summary_path,
            runner_args=runner_args,
        )

        t0 = perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = perf_counter() - t0
        stdout_log = log_root / f"{idx:02d}_{name}.stdout"
        stderr_log = log_root / f"{idx:02d}_{name}.stderr"
        stdout_log.write_text(proc.stdout, encoding="utf-8")
        stderr_log.write_text(proc.stderr, encoding="utf-8")

        report = _read_json(summary_path)
        runner_passed = bool(proc.returncode == 0 and isinstance(report, dict) and report.get("passed", False))
        eval_failures: list[str] = []
        eval_snapshot: dict[str, float] = {}

        if not runner_passed:
            eval_failures.append("runner_failed")
            if isinstance(report, dict) and isinstance(report.get("failures"), dict):
                for key in report["failures"]:
                    eval_failures.append(f"runner:{key}")

        expected = dict(expected_positive if mode == "positive" else expected_negative)
        case_expected = case_raw.get("expected", {})
        if isinstance(case_expected, dict):
            expected = _merge(expected, case_expected)

        if isinstance(report, dict):
            if mode == "positive":
                ok_eval, checks, snap = _eval_positive(report, expected)
            else:
                ok_eval, checks, snap = _eval_negative(report, expected)
            if not ok_eval:
                eval_failures.extend(checks)
            eval_snapshot = snap
        else:
            eval_failures.append("summary_missing_or_invalid")

        case_passed = runner_passed and len(eval_failures) == 0
        records.append(
            {
                "name": name,
                "mode": mode,
                "enabled": True,
                "passed": case_passed,
                "runner_passed": runner_passed,
                "returncode": int(proc.returncode),
                "duration_s": dt,
                "command": cmd,
                "resolved_config": str(resolved_cfg_path),
                "summary_json": str(summary_path),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "expected": expected,
                "evaluation_failures": eval_failures,
                "metrics_snapshot": eval_snapshot,
                "summary": report,
            }
        )

    finished = datetime.now()
    total = len(records)
    passed_count = int(sum(1 for row in records if bool(row.get("passed", False))))
    failed_names = [str(row.get("name")) for row in records if not bool(row.get("passed", False))]
    all_passed = passed_count == total and total > 0
    overall = total > 0 and passed_count >= min_pass_count and (all_passed if require_all else True)

    payload = {
        "config": str(args.config),
        "out": str(out_path),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "duration_s": (finished - started).total_seconds(),
        "config_require_all": config_require_all,
        "config_min_pass_count": config_min_pass_count,
        "require_all": require_all,
        "min_pass_count": min_pass_count,
        "case_total": total,
        "passed_count": passed_count,
        "failed_names": failed_names,
        "passed": overall,
        "cases": records,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
