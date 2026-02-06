"""
Run `run_virtual_cycles` from a YAML config file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # type: ignore

from sim.tests.virtual_cycle import run_virtual_cycles

PATH_KEYS = {
    "csv_output",
    "analysis_csv",
    "data_output",
    "dump_dir",
    "vtk_dir",
    "initial_vtk",
    "stress_strain_csv",
    "run_dir",
}
TUPLE_KEYS = {
    "orientation_vector",
    "grid_shape",
    "grid_spacing",
    "grid_periodic",
    "stable_metrics",
}


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
        msg = str(item.message)
        src = f"{Path(item.filename).name}:{item.lineno}"
        key = f"{src} | {item.category.__name__}: {msg}"
        counts[key] = counts.get(key, 0) + 1
    return counts


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


def _normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    for key in PATH_KEYS:
        if key in out and out[key] is not None:
            out[key] = Path(out[key])
    for key in TUPLE_KEYS:
        if key in out and isinstance(out[key], list):
            out[key] = tuple(out[key])
    if "notch_box" in out and isinstance(out["notch_box"], list):
        out["notch_box"] = tuple(tuple(row) for row in out["notch_box"])
    if "grid_shape" in out and out["grid_shape"] is not None:
        out["grid_shape"] = tuple(int(v) for v in out["grid_shape"])
    if "grid_periodic" in out and out["grid_periodic"] is not None:
        out["grid_periodic"] = tuple(bool(v) for v in out["grid_periodic"])
    if "grid_spacing" in out and out["grid_spacing"] is not None:
        out["grid_spacing"] = tuple(float(v) for v in out["grid_spacing"])
    if "orientation_vector" in out and out["orientation_vector"] is not None:
        out["orientation_vector"] = tuple(float(v) for v in out["orientation_vector"])
    return out


def _resolve_payload(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "virtual_cycle" in raw:
        vc = raw.get("virtual_cycle")
        if not isinstance(vc, dict):
            raise ValueError("virtual_cycle must be a mapping.")
        meta = {k: v for k, v in raw.items() if k != "virtual_cycle"}
        return vc, meta
    return raw, {}


def _resolve_summary_dir(cfg: dict[str, Any]) -> Path | None:
    for key in ("run_dir", "csv_output", "analysis_csv", "stress_strain_csv"):
        val = cfg.get(key)
        if val is None:
            continue
        p = Path(val)
        return p if key == "run_dir" else p.parent
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run virtual_cycle from YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional summary JSON output path.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved kwargs and exit.")
    parser.add_argument(
        "--max-runtime-warnings",
        type=int,
        default=None,
        help="Fail run when RuntimeWarning count exceeds this threshold.",
    )
    parser.add_argument(
        "--allow-nonfinite-last-cycle",
        action="store_true",
        help="Do not fail when last_cycle contains NaN/Inf.",
    )
    parser.add_argument(
        "--max-mechanical-cg-failures",
        type=int,
        default=None,
        help="Fail when stability_diagnostics.mechanical_cg_failures exceeds this threshold.",
    )
    parser.add_argument(
        "--max-mechanical-not-accepted-steps",
        type=int,
        default=None,
        help="Fail when stability_diagnostics.mechanical_not_accepted_steps exceeds this threshold.",
    )
    parser.add_argument(
        "--max-crack-cg-nonconverged-steps",
        type=int,
        default=None,
        help="Fail when stability_diagnostics.crack_cg_nonconverged_steps exceeds this threshold.",
    )
    parser.add_argument(
        "--max-nonfinite-count",
        type=int,
        default=0,
        help="Fail when stability_diagnostics.nonfinite_count exceeds this threshold.",
    )
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")

    vc_cfg_raw, meta = _resolve_payload(raw)
    cfg = _normalize_config(vc_cfg_raw)

    if args.dry_run:
        print(json.dumps({"config": _to_builtin(cfg), "meta": _to_builtin(meta)}, indent=2))
        return 0

    t0 = datetime.now()
    run_cfg = dict(cfg)
    run_cfg.pop("diagnostics_out", None)
    diagnostics: dict[str, float | int | bool] = {}
    results = []
    paris_coeff = 0.0
    coffman = 0.0
    error: dict[str, Any] | None = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        try:
            results, paris_coeff, coffman = run_virtual_cycles(**run_cfg, diagnostics_out=diagnostics)
        except Exception as exc:  # pragma: no cover - defensive summary path
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc().splitlines()[-12:],
            }
        warning_counts = _runtime_warning_counts(caught)
    t1 = datetime.now()

    last = results[-1] if results else None
    last_cycle = None if last is None else _to_builtin(last.__dict__)
    last_cycle_finite = True if last_cycle is None else _is_finite_tree(last_cycle)
    diagnostics_finite = _is_finite_tree(diagnostics)
    runtime_warning_count = int(sum(warning_counts.values()))
    warning_items = [
        {"message": msg, "count": cnt}
        for msg, cnt in sorted(warning_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    failure_reasons: list[str] = []
    if error is not None:
        failure_reasons.append("runtime_exception")
    if not args.allow_nonfinite_last_cycle and not last_cycle_finite:
        failure_reasons.append("last_cycle_nonfinite")
    if not diagnostics_finite:
        failure_reasons.append("diagnostics_nonfinite")
    if args.max_runtime_warnings is not None and runtime_warning_count > args.max_runtime_warnings:
        failure_reasons.append(
            f"runtime_warning_count_exceeded({runtime_warning_count}>{args.max_runtime_warnings})"
        )
    mech_cg_failures = int(diagnostics.get("mechanical_cg_failures", 0))
    mech_not_accepted_steps = int(diagnostics.get("mechanical_not_accepted_steps", 0))
    crack_cg_nonconverged_steps = int(diagnostics.get("crack_cg_nonconverged_steps", 0))
    nonfinite_count = int(diagnostics.get("nonfinite_count", 0))
    if args.max_mechanical_cg_failures is not None and mech_cg_failures > args.max_mechanical_cg_failures:
        failure_reasons.append(
            f"mechanical_cg_failures_exceeded({mech_cg_failures}>{args.max_mechanical_cg_failures})"
        )
    if (
        args.max_mechanical_not_accepted_steps is not None
        and mech_not_accepted_steps > args.max_mechanical_not_accepted_steps
    ):
        failure_reasons.append(
            "mechanical_not_accepted_steps_exceeded("
            f"{mech_not_accepted_steps}>{args.max_mechanical_not_accepted_steps})"
        )
    if (
        args.max_crack_cg_nonconverged_steps is not None
        and crack_cg_nonconverged_steps > args.max_crack_cg_nonconverged_steps
    ):
        failure_reasons.append(
            "crack_cg_nonconverged_steps_exceeded("
            f"{crack_cg_nonconverged_steps}>{args.max_crack_cg_nonconverged_steps})"
        )
    if nonfinite_count > args.max_nonfinite_count:
        failure_reasons.append(f"nonfinite_count_exceeded({nonfinite_count}>{args.max_nonfinite_count})")
    passed = len(failure_reasons) == 0

    summary = {
        "config_path": str(args.config),
        "started_at": t0.isoformat(timespec="seconds"),
        "finished_at": t1.isoformat(timespec="seconds"),
        "duration_s": (t1 - t0).total_seconds(),
        "meta": _to_builtin(meta),
        "config": _to_builtin(cfg),
        "passed": passed,
        "failure_reasons": failure_reasons,
        "error": error,
        "runtime_warning_count": runtime_warning_count,
        "runtime_warning_items": warning_items,
        "limits": {
            "max_runtime_warnings": args.max_runtime_warnings,
            "max_mechanical_cg_failures": args.max_mechanical_cg_failures,
            "max_mechanical_not_accepted_steps": args.max_mechanical_not_accepted_steps,
            "max_crack_cg_nonconverged_steps": args.max_crack_cg_nonconverged_steps,
            "max_nonfinite_count": args.max_nonfinite_count,
        },
        "stability_diagnostics": _to_builtin(diagnostics),
        "checks": {
            "last_cycle_finite": last_cycle_finite,
            "diagnostics_finite": diagnostics_finite,
        },
        "cycles_completed": len(results),
        "paris_coeff": float(paris_coeff) if math.isfinite(float(paris_coeff)) else None,
        "coffman_coeff": float(coffman) if math.isfinite(float(coffman)) else None,
        "last_cycle": last_cycle,
    }

    out_dir = _resolve_summary_dir(cfg)
    summary_path = args.summary_output
    if summary_path is None:
        if out_dir is not None:
            summary_path = out_dir / "run_summary.json"
        else:
            day = date.today().isoformat()
            ts = datetime.now().strftime("%H%M%S")
            summary_path = Path("sim/tests/runs") / day / f"config_run_{args.config.stem}_{ts}" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)

    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
