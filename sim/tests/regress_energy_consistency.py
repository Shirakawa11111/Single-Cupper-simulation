"""
Energy-consistency regression gate for fatigue virtual-cycle runs.

Default workflow:
1) Run virtual-cycle simulation from a YAML config (or read an existing CSV).
2) Check cycle-level energy/crack/plastic trends.
3) Fail on nonphysical reversals or stability threshold violations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.tests.run_virtual_cycle_config import _normalize_config, _resolve_payload
from sim.tests.virtual_cycle import run_virtual_cycles


@dataclass
class Thresholds:
    min_cycles: int = 5
    max_energy_drop_count: int = 0
    energy_drop_tol: float = 1e-10
    max_crack_reversal_count: int = 0
    crack_reversal_tol: float = 1e-12
    max_plastic_reversal_count: int = 0
    plastic_reversal_tol: float = 1e-12
    max_negative_crack_delta_count: int = 0
    crack_delta_neg_tol: float = 1e-12
    min_crack_slope: float = 0.0
    min_accum_plastic_slope: float = 0.0
    min_plastic_range_median: float = 0.0
    max_runtime_warnings: int = 50
    max_mechanical_not_accepted_steps: int = 160
    max_crack_cg_nonconverged_steps: int = 40
    max_nonfinite_count: int = 0


@dataclass
class Metrics:
    n_cycles: int
    energy_drop_count: int
    crack_reversal_count: int
    plastic_reversal_count: int
    negative_crack_delta_count: int
    crack_slope_per_cycle: float
    accum_plastic_slope_per_cycle: float
    crack_total_increase: float
    accum_plastic_total_increase: float
    energy_total_change: float
    plastic_range_median: float
    min_crack_delta: float


@dataclass
class Report:
    passed: bool
    failures: dict[str, str]
    thresholds: Thresholds
    metrics: Metrics
    run: dict[str, Any]
    timing: dict[str, float]


def _default_out() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"energy_gate_{ts}" / "summary.json"


def _default_run_dir() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"energy_gate_run_{ts}"


def _runtime_warning_count(caught: list[warnings.WarningMessage]) -> int:
    total = 0
    for item in caught:
        if issubclass(item.category, RuntimeWarning):
            total += 1
    return total


def _prepare_runtime_cfg(config_path: Path, run_dir: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")
    vc_raw, _meta = _resolve_payload(raw)
    cfg = _normalize_config(vc_raw)
    cfg["run_dir"] = run_dir
    cfg["auto_output"] = True
    for key in (
        "csv_output",
        "analysis_csv",
        "data_output",
        "dump_dir",
        "vtk_dir",
        "initial_vtk",
        "stress_strain_csv",
    ):
        cfg.pop(key, None)
    return cfg


def _load_cycle_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"Cycle CSV is empty: {path}")

    required = ("cycle", "energy", "crack_mean", "accum_plastic_mean", "plastic_range")
    for col in required:
        if col not in rows[0]:
            raise ValueError(f"Cycle CSV missing required column: {col}")

    def _arr(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows], dtype=float)

    out = {
        "cycle": _arr("cycle"),
        "energy": _arr("energy"),
        "crack_mean": _arr("crack_mean"),
        "accum_plastic_mean": _arr("accum_plastic_mean"),
        "plastic_range": _arr("plastic_range"),
    }
    if "crack_delta" in rows[0]:
        out["crack_delta"] = _arr("crack_delta")
    else:
        out["crack_delta"] = np.full(len(rows), np.nan, dtype=float)
    return out


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    den = float(np.sum((x - x_mean) ** 2))
    if den <= 1e-30:
        return 0.0
    num = float(np.sum((x - x_mean) * (y - y_mean)))
    return num / den


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy-consistency regression gate.")
    parser.add_argument("--config", type=Path, default=Path("sim/configs/fatigue_lowamp_align_locked_v4.yaml"))
    parser.add_argument("--virtual-cycle-csv", type=Path, default=None, help="Use existing cycle CSV and skip runtime.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory when simulation is executed.")
    parser.add_argument("--out", type=Path, default=None, help="Output summary JSON path.")
    parser.add_argument("--min-cycles", type=int, default=5)
    parser.add_argument("--max-energy-drop-count", type=int, default=0)
    parser.add_argument("--energy-drop-tol", type=float, default=1e-10)
    parser.add_argument("--max-crack-reversal-count", type=int, default=0)
    parser.add_argument("--crack-reversal-tol", type=float, default=1e-12)
    parser.add_argument("--max-plastic-reversal-count", type=int, default=0)
    parser.add_argument("--plastic-reversal-tol", type=float, default=1e-12)
    parser.add_argument("--max-negative-crack-delta-count", type=int, default=0)
    parser.add_argument("--crack-delta-neg-tol", type=float, default=1e-12)
    parser.add_argument("--min-crack-slope", type=float, default=0.0)
    parser.add_argument("--min-accum-plastic-slope", type=float, default=0.0)
    parser.add_argument("--min-plastic-range-median", type=float, default=0.0)
    parser.add_argument("--max-runtime-warnings", type=int, default=50)
    parser.add_argument("--max-mechanical-not-accepted-steps", type=int, default=160)
    parser.add_argument("--max-crack-cg-nonconverged-steps", type=int, default=40)
    parser.add_argument("--max-nonfinite-count", type=int, default=0)
    args = parser.parse_args()

    out_path = args.out or _default_out()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    thresholds = Thresholds(
        min_cycles=args.min_cycles,
        max_energy_drop_count=args.max_energy_drop_count,
        energy_drop_tol=args.energy_drop_tol,
        max_crack_reversal_count=args.max_crack_reversal_count,
        crack_reversal_tol=args.crack_reversal_tol,
        max_plastic_reversal_count=args.max_plastic_reversal_count,
        plastic_reversal_tol=args.plastic_reversal_tol,
        max_negative_crack_delta_count=args.max_negative_crack_delta_count,
        crack_delta_neg_tol=args.crack_delta_neg_tol,
        min_crack_slope=args.min_crack_slope,
        min_accum_plastic_slope=args.min_accum_plastic_slope,
        min_plastic_range_median=args.min_plastic_range_median,
        max_runtime_warnings=args.max_runtime_warnings,
        max_mechanical_not_accepted_steps=args.max_mechanical_not_accepted_steps,
        max_crack_cg_nonconverged_steps=args.max_crack_cg_nonconverged_steps,
        max_nonfinite_count=args.max_nonfinite_count,
    )

    t0 = perf_counter()
    run_info: dict[str, Any] = {}
    diag: dict[str, Any] = {}
    runtime_warning_count = 0

    if args.virtual_cycle_csv is not None:
        cycle_csv = args.virtual_cycle_csv
        run_info["mode"] = "external_csv"
        run_info["virtual_cycle_csv"] = str(cycle_csv)
        run_info["runtime_warning_count"] = None
        run_info["stability_diagnostics"] = None
    else:
        run_dir = args.run_dir or _default_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = _prepare_runtime_cfg(args.config, run_dir)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            results, paris, coffman = run_virtual_cycles(**cfg, diagnostics_out=diag)
            runtime_warning_count = _runtime_warning_count(caught)
        cycle_csv = run_dir / "virtual_cycle.csv"
        run_info.update(
            {
                "mode": "run_from_config",
                "config": str(args.config),
                "run_dir": str(run_dir),
                "virtual_cycle_csv": str(cycle_csv),
                "cycles_completed": len(results),
                "paris_coeff": float(paris) if math.isfinite(float(paris)) else None,
                "coffman_coeff": float(coffman) if math.isfinite(float(coffman)) else None,
                "runtime_warning_count": runtime_warning_count,
                "stability_diagnostics": diag,
            }
        )

    data = _load_cycle_csv(cycle_csv)
    cycle = data["cycle"]
    energy = data["energy"]
    crack_mean = data["crack_mean"]
    accum_plastic = data["accum_plastic_mean"]
    plastic_range = data["plastic_range"]
    crack_delta = data["crack_delta"]

    n_cycles = int(cycle.size)
    energy_diff = np.diff(energy)
    crack_diff = np.diff(crack_mean)
    plastic_diff = np.diff(accum_plastic)

    energy_drop_count = int(np.sum(energy_diff < -thresholds.energy_drop_tol))
    crack_reversal_count = int(np.sum(crack_diff < -thresholds.crack_reversal_tol))
    plastic_reversal_count = int(np.sum(plastic_diff < -thresholds.plastic_reversal_tol))
    negative_crack_delta_count = int(np.sum(crack_delta < -thresholds.crack_delta_neg_tol))

    crack_slope = _linear_slope(cycle, crack_mean)
    plastic_slope = _linear_slope(cycle, accum_plastic)

    crack_total_inc = float(crack_mean[-1] - crack_mean[0]) if n_cycles > 0 else 0.0
    plastic_total_inc = float(accum_plastic[-1] - accum_plastic[0]) if n_cycles > 0 else 0.0
    energy_total_change = float(energy[-1] - energy[0]) if n_cycles > 0 else 0.0
    plastic_range_median = float(np.median(plastic_range)) if n_cycles > 0 else 0.0
    min_crack_delta = float(np.min(crack_delta)) if n_cycles > 0 else 0.0

    metrics = Metrics(
        n_cycles=n_cycles,
        energy_drop_count=energy_drop_count,
        crack_reversal_count=crack_reversal_count,
        plastic_reversal_count=plastic_reversal_count,
        negative_crack_delta_count=negative_crack_delta_count,
        crack_slope_per_cycle=crack_slope,
        accum_plastic_slope_per_cycle=plastic_slope,
        crack_total_increase=crack_total_inc,
        accum_plastic_total_increase=plastic_total_inc,
        energy_total_change=energy_total_change,
        plastic_range_median=plastic_range_median,
        min_crack_delta=min_crack_delta,
    )

    failures: dict[str, str] = {}
    if n_cycles < thresholds.min_cycles:
        failures["min_cycles"] = f"{n_cycles} < {thresholds.min_cycles}"

    finite_ok = bool(
        np.all(np.isfinite(energy))
        and np.all(np.isfinite(crack_mean))
        and np.all(np.isfinite(accum_plastic))
        and np.all(np.isfinite(plastic_range))
        and np.all(np.isfinite(crack_delta))
    )
    if not finite_ok:
        failures["finite"] = "non-finite values in cycle metrics"

    if energy_drop_count > thresholds.max_energy_drop_count:
        failures["energy_drop_count"] = (
            f"{energy_drop_count} > {thresholds.max_energy_drop_count}"
        )
    if crack_reversal_count > thresholds.max_crack_reversal_count:
        failures["crack_reversal_count"] = (
            f"{crack_reversal_count} > {thresholds.max_crack_reversal_count}"
        )
    if plastic_reversal_count > thresholds.max_plastic_reversal_count:
        failures["plastic_reversal_count"] = (
            f"{plastic_reversal_count} > {thresholds.max_plastic_reversal_count}"
        )
    if negative_crack_delta_count > thresholds.max_negative_crack_delta_count:
        failures["negative_crack_delta_count"] = (
            f"{negative_crack_delta_count} > {thresholds.max_negative_crack_delta_count}"
        )
    if crack_slope < thresholds.min_crack_slope:
        failures["crack_slope_per_cycle"] = f"{crack_slope:.6e} < {thresholds.min_crack_slope:.6e}"
    if plastic_slope < thresholds.min_accum_plastic_slope:
        failures["accum_plastic_slope_per_cycle"] = (
            f"{plastic_slope:.6e} < {thresholds.min_accum_plastic_slope:.6e}"
        )
    if plastic_range_median < thresholds.min_plastic_range_median:
        failures["plastic_range_median"] = (
            f"{plastic_range_median:.6e} < {thresholds.min_plastic_range_median:.6e}"
        )

    if args.virtual_cycle_csv is None:
        if runtime_warning_count > thresholds.max_runtime_warnings:
            failures["runtime_warning_count"] = (
                f"{runtime_warning_count} > {thresholds.max_runtime_warnings}"
            )
        mech_na = int(diag.get("mechanical_not_accepted_steps", 0))
        crack_cg = int(diag.get("crack_cg_nonconverged_steps", 0))
        nonfinite = int(diag.get("nonfinite_count", 0))
        if mech_na > thresholds.max_mechanical_not_accepted_steps:
            failures["mechanical_not_accepted_steps"] = (
                f"{mech_na} > {thresholds.max_mechanical_not_accepted_steps}"
            )
        if crack_cg > thresholds.max_crack_cg_nonconverged_steps:
            failures["crack_cg_nonconverged_steps"] = (
                f"{crack_cg} > {thresholds.max_crack_cg_nonconverged_steps}"
            )
        if nonfinite > thresholds.max_nonfinite_count:
            failures["nonfinite_count"] = f"{nonfinite} > {thresholds.max_nonfinite_count}"

    t1 = perf_counter()
    report = Report(
        passed=len(failures) == 0,
        failures=failures,
        thresholds=thresholds,
        metrics=metrics,
        run=run_info,
        timing={"total_s": t1 - t0},
    )
    payload = json.dumps(asdict(report), indent=2)
    out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

