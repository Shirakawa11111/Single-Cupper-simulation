"""
Scan candidate loading/defect configurations and detect crack-onset cases.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.tests.virtual_cycle import run_virtual_cycles


def _default_out_dir() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/runs") / day / f"crack_onset_scan_{ts}"


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    return value


def _is_finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_is_finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_finite_tree(v) for v in value)
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _runtime_warning_counts(caught: list[warnings.WarningMessage]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in caught:
        if not issubclass(item.category, RuntimeWarning):
            continue
        src = f"{Path(item.filename).name}:{item.lineno}"
        key = f"{src} | {item.category.__name__}: {item.message}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)
        else:
            out[key] = val
    return out


def _sanitize_task(name: str) -> str:
    chars: list[str] = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars) or "case"


def _normalize_vc_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    path_keys = {
        "csv_output",
        "analysis_csv",
        "data_output",
        "dump_dir",
        "vtk_dir",
        "initial_vtk",
        "stress_strain_csv",
        "run_dir",
    }
    for key in path_keys:
        if key in out and out[key] is not None:
            out[key] = Path(out[key])
    if "notch_box" in out and isinstance(out["notch_box"], list):
        out["notch_box"] = tuple(tuple(row) for row in out["notch_box"])
    if "grid_shape" in out and out["grid_shape"] is not None:
        out["grid_shape"] = tuple(int(v) for v in out["grid_shape"])
    if "grid_spacing" in out and out["grid_spacing"] is not None:
        out["grid_spacing"] = tuple(float(v) for v in out["grid_spacing"])
    if "grid_periodic" in out and out["grid_periodic"] is not None:
        out["grid_periodic"] = tuple(bool(v) for v in out["grid_periodic"])
    if "orientation_vector" in out and out["orientation_vector"] is not None:
        out["orientation_vector"] = tuple(float(v) for v in out["orientation_vector"])
    if "stable_metrics" in out and isinstance(out["stable_metrics"], list):
        out["stable_metrics"] = tuple(str(v) for v in out["stable_metrics"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan crack-onset candidates from YAML config.")
    parser.add_argument("--config", type=Path, default=Path("sim/configs/crack_onset_scan.yaml"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=None, help="Optional cap for quick dry runs.")
    parser.add_argument(
        "--max-runtime-warnings",
        type=int,
        default=None,
        help="Override criteria.max_runtime_warnings from YAML.",
    )
    parser.add_argument(
        "--min-onset-cases",
        type=int,
        default=None,
        help="Override criteria.min_onset_cases from YAML.",
    )
    parser.add_argument(
        "--max-mechanical-cg-failures",
        type=int,
        default=None,
        help="Override criteria.max_mechanical_cg_failures from YAML.",
    )
    parser.add_argument(
        "--max-mechanical-not-accepted-steps",
        type=int,
        default=None,
        help="Override criteria.max_mechanical_not_accepted_steps from YAML.",
    )
    parser.add_argument(
        "--max-crack-cg-nonconverged-steps",
        type=int,
        default=None,
        help="Override criteria.max_crack_cg_nonconverged_steps from YAML.",
    )
    parser.add_argument(
        "--max-nonfinite-count",
        type=int,
        default=None,
        help="Override criteria.max_nonfinite_count from YAML.",
    )
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")
    defaults = raw.get("defaults", {}) or {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping.")
    default_vc = defaults.get("virtual_cycle", {}) or {}
    if not isinstance(default_vc, dict):
        raise ValueError("defaults.virtual_cycle must be a mapping.")
    criteria = defaults.get("criteria", {}) or {}
    if not isinstance(criteria, dict):
        raise ValueError("defaults.criteria must be a mapping.")

    min_crack_delta = float(criteria.get("min_crack_delta", 1e-4))
    min_crack_mean_delta = float(criteria.get("min_crack_mean_delta", 1e-4))
    allow_mean_aux = bool(criteria.get("allow_mean_aux", True))
    min_crack_length_for_mean_aux = float(criteria.get("min_crack_length_for_mean_aux", min_crack_delta))
    max_runtime_warnings = criteria.get("max_runtime_warnings", None)
    max_runtime_warnings = None if max_runtime_warnings is None else int(max_runtime_warnings)
    min_onset_cases = int(criteria.get("min_onset_cases", 1))
    max_mechanical_cg_failures = criteria.get("max_mechanical_cg_failures", None)
    max_mechanical_cg_failures = (
        None if max_mechanical_cg_failures is None else int(max_mechanical_cg_failures)
    )
    max_mechanical_not_accepted_steps = criteria.get("max_mechanical_not_accepted_steps", None)
    max_mechanical_not_accepted_steps = (
        None if max_mechanical_not_accepted_steps is None else int(max_mechanical_not_accepted_steps)
    )
    max_crack_cg_nonconverged_steps = criteria.get("max_crack_cg_nonconverged_steps", None)
    max_crack_cg_nonconverged_steps = (
        None if max_crack_cg_nonconverged_steps is None else int(max_crack_cg_nonconverged_steps)
    )
    max_nonfinite_count = int(criteria.get("max_nonfinite_count", 0))
    if args.max_runtime_warnings is not None:
        max_runtime_warnings = int(args.max_runtime_warnings)
    if args.min_onset_cases is not None:
        min_onset_cases = int(args.min_onset_cases)
    if args.max_mechanical_cg_failures is not None:
        max_mechanical_cg_failures = int(args.max_mechanical_cg_failures)
    if args.max_mechanical_not_accepted_steps is not None:
        max_mechanical_not_accepted_steps = int(args.max_mechanical_not_accepted_steps)
    if args.max_crack_cg_nonconverged_steps is not None:
        max_crack_cg_nonconverged_steps = int(args.max_crack_cg_nonconverged_steps)
    if args.max_nonfinite_count is not None:
        max_nonfinite_count = int(args.max_nonfinite_count)

    cases = raw.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list.")
    if args.max_cases is not None:
        cases = cases[: max(0, args.max_cases)]

    out_dir = args.out or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now()
    case_rows: list[dict[str, Any]] = []
    for idx, case_raw in enumerate(cases, start=1):
        if not isinstance(case_raw, dict):
            raise ValueError(f"case #{idx} must be a mapping.")
        name = str(case_raw.get("name", f"case_{idx:02d}"))
        overrides = case_raw.get("virtual_cycle", {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"case `{name}` virtual_cycle must be a mapping.")
        vc_cfg = _merge(default_vc, overrides)
        case_key = _sanitize_task(name)
        case_dir = out_dir / case_key
        case_dir.mkdir(parents=True, exist_ok=True)

        vc_cfg = dict(vc_cfg)
        vc_cfg["task"] = f"scan_{case_key}"
        vc_cfg["run_dir"] = case_dir
        vc_cfg["auto_output"] = True
        vc_cfg = _normalize_vc_config(vc_cfg)

        diagnostics: dict[str, float | int | bool] = {}
        error: dict[str, Any] | None = None
        results = []
        paris_coeff = 0.0
        coffman = 0.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            try:
                results, paris_coeff, coffman = run_virtual_cycles(**vc_cfg, diagnostics_out=diagnostics)
            except Exception as exc:  # pragma: no cover - defensive summary path
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines()[-12:],
                }
            warning_counts = _runtime_warning_counts(caught)

        first = results[0] if results else None
        last = results[-1] if results else None
        crack_len0 = float(first.crack_length) if first is not None else 0.0
        crack_len1 = float(last.crack_length) if last is not None else 0.0
        crack_mean0 = float(first.crack_mean) if first is not None else 0.0
        crack_mean1 = float(last.crack_mean) if last is not None else 0.0
        crack_delta = crack_len1 - crack_len0
        crack_mean_delta = crack_mean1 - crack_mean0
        runtime_warning_count = int(sum(warning_counts.values()))
        mech_failures = int(diagnostics.get("mechanical_cg_failures", 0))
        mech_not_accepted_steps = int(diagnostics.get("mechanical_not_accepted_steps", 0))
        crack_cg_nonconv = int(diagnostics.get("crack_cg_nonconverged_steps", 0))
        nonfinite_count = int(diagnostics.get("nonfinite_count", 0))
        finite_ok = _is_finite_tree(diagnostics) and _is_finite_tree(
            None if last is None else last.__dict__
        )
        warning_ok = True if max_runtime_warnings is None else runtime_warning_count <= max_runtime_warnings
        mech_ok = True if max_mechanical_cg_failures is None else mech_failures <= max_mechanical_cg_failures
        mech_accept_ok = (
            True
            if max_mechanical_not_accepted_steps is None
            else mech_not_accepted_steps <= max_mechanical_not_accepted_steps
        )
        crack_cg_ok = (
            True
            if max_crack_cg_nonconverged_steps is None
            else crack_cg_nonconv <= max_crack_cg_nonconverged_steps
        )
        nonfinite_ok = nonfinite_count <= max_nonfinite_count
        onset_length = crack_delta >= min_crack_delta
        onset_mean_aux = crack_mean_delta >= min_crack_mean_delta and crack_len1 >= min_crack_length_for_mean_aux
        onset = onset_length or (allow_mean_aux and onset_mean_aux)
        checks_ok = (
            error is None
            and finite_ok
            and warning_ok
            and mech_ok
            and mech_accept_ok
            and crack_cg_ok
            and nonfinite_ok
        )
        passed = (
            checks_ok
            and onset
        )

        row = {
            "name": name,
            "task": case_key,
            "passed": passed,
            "onset": onset,
            "onset_length": onset_length,
            "onset_mean_aux": onset_mean_aux,
            "checks_ok": checks_ok,
            "error": error,
            "cycles_completed": len(results),
            "crack_length_initial": crack_len0,
            "crack_length_final": crack_len1,
            "crack_delta": crack_delta,
            "crack_mean_initial": crack_mean0,
            "crack_mean_final": crack_mean1,
            "crack_mean_delta": crack_mean_delta,
            "runtime_warning_count": runtime_warning_count,
            "runtime_warning_items": [
                {"message": msg, "count": cnt}
                for msg, cnt in sorted(warning_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
            "finite_ok": finite_ok,
            "warning_ok": warning_ok,
            "mech_ok": mech_ok,
            "mech_accept_ok": mech_accept_ok,
            "crack_cg_ok": crack_cg_ok,
            "nonfinite_ok": nonfinite_ok,
            "mech_not_accepted_steps": mech_not_accepted_steps,
            "paris_coeff": float(paris_coeff) if math.isfinite(float(paris_coeff)) else None,
            "coffman_coeff": float(coffman) if math.isfinite(float(coffman)) else None,
            "stability_diagnostics": _to_builtin(diagnostics),
            "config": _to_builtin(vc_cfg),
        }
        case_rows.append(row)
        (case_dir / "case_summary.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")

    onset_cases = int(sum(1 for r in case_rows if r.get("onset", False)))
    checks_passed = all(
        (
            bool(r.get("checks_ok", False))
        )
        for r in case_rows
    )
    passed = onset_cases >= min_onset_cases and checks_passed
    failure_reasons: list[str] = []
    if onset_cases < min_onset_cases:
        failure_reasons.append(f"onset_cases_insufficient({onset_cases}<{min_onset_cases})")
    if not checks_passed:
        failed = [str(r.get("name", "")) for r in case_rows if not bool(r.get("checks_ok", False))]
        failure_reasons.append(f"case_checks_failed({','.join(failed)})")

    finished_at = datetime.now()
    summary = {
        "config_path": str(args.config),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_s": (finished_at - started_at).total_seconds(),
        "passed": passed,
        "failure_reasons": failure_reasons,
        "criteria": {
            "min_crack_delta": min_crack_delta,
            "min_crack_mean_delta": min_crack_mean_delta,
            "allow_mean_aux": allow_mean_aux,
            "min_crack_length_for_mean_aux": min_crack_length_for_mean_aux,
            "max_runtime_warnings": max_runtime_warnings,
            "min_onset_cases": min_onset_cases,
            "max_mechanical_cg_failures": max_mechanical_cg_failures,
            "max_mechanical_not_accepted_steps": max_mechanical_not_accepted_steps,
            "max_crack_cg_nonconverged_steps": max_crack_cg_nonconverged_steps,
            "max_nonfinite_count": max_nonfinite_count,
        },
        "cases_total": len(case_rows),
        "onset_cases": onset_cases,
        "checks_passed": checks_passed,
        "cases": case_rows,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "name,passed,onset,onset_length,onset_mean_aux,cycles_completed,crack_delta,crack_mean_delta,"
            "crack_length_final,crack_mean_final,runtime_warning_count,"
            "finite_ok,warning_ok,mech_ok,mech_accept_ok,crack_cg_ok,nonfinite_ok,mech_not_accepted_steps\n"
        )
        for r in case_rows:
            fh.write(
                f"{r['name']},{int(bool(r['passed']))},{int(bool(r['onset']))},"
                f"{int(bool(r['onset_length']))},{int(bool(r['onset_mean_aux']))},"
                f"{int(r['cycles_completed'])},{float(r['crack_delta']):.6e},"
                f"{float(r['crack_mean_delta']):.6e},{float(r['crack_length_final']):.6e},"
                f"{float(r['crack_mean_final']):.6e},{int(r['runtime_warning_count'])},"
                f"{int(bool(r['finite_ok']))},{int(bool(r['warning_ok']))},"
                f"{int(bool(r['mech_ok']))},{int(bool(r['mech_accept_ok']))},"
                f"{int(bool(r['crack_cg_ok']))},{int(bool(r['nonfinite_ok']))},"
                f"{int(r['mech_not_accepted_steps'])}\n"
            )

    print(json.dumps({"summary_path": str(summary_path), "csv_path": str(csv_path), "passed": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
