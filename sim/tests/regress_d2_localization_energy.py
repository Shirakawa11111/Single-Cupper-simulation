"""
D2 regression gate: crack localization trigger + energy-density field export.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.tests.run_virtual_cycle_config import _normalize_config, _resolve_payload
from sim.tests.virtual_cycle import CycleResult, run_virtual_cycles

import yaml  # type: ignore


@dataclass
class Thresholds:
    min_cycles: int = 3
    min_crack_delta: float = 5.0e-2
    min_localization_index: float = 3.0
    min_energy_crack_mean: float = 1.0e-10
    min_energy_total_density_mean: float = 1.0e-10
    max_runtime_warnings: int = 50
    max_mechanical_not_accepted_steps: int = 160
    max_crack_cg_nonconverged_steps: int = 20
    max_nonfinite_count: int = 0
    min_vtk_energy_fields: int = 4


@dataclass
class Metrics:
    cycles_completed: int
    crack_mean_initial: float
    crack_mean_final: float
    crack_delta_total: float
    crack_localization_index_final: float
    crack_localization_index_peak: float
    energy_crack_mean_final: float
    energy_total_density_mean_final: float
    runtime_warning_count: int
    mechanical_not_accepted_steps: int
    crack_cg_nonconverged_steps: int
    nonfinite_count: int
    vtk_energy_field_count: int
    vtk_energy_fields_present: list[str]


@dataclass
class Report:
    passed: bool
    failures: dict[str, str]
    thresholds: Thresholds
    metrics: Metrics
    outputs: dict[str, Any]
    run: dict[str, Any]
    timing: dict[str, float]


def _default_out() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"d2_localization_energy_{ts}" / "summary.json"


def _runtime_warning_count(caught: list[warnings.WarningMessage]) -> int:
    total = 0
    for item in caught:
        if issubclass(item.category, RuntimeWarning):
            total += 1
    return total


def _extract_vtk_scalars(vtk_path: Path) -> list[str]:
    blob = vtk_path.read_bytes()
    names = [m.decode("ascii", errors="ignore") for m in re.findall(rb"SCALARS\s+([A-Za-z0-9_\-]+)\s+float\s+1", blob)]
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def _read_csv_header(path: Path) -> list[str]:
    first = path.read_text(encoding="utf-8").splitlines()[0] if path.exists() else ""
    return [item.strip() for item in first.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run D2 localization + energy-density gate.")
    parser.add_argument("--config", type=Path, default=Path("sim/configs/d2_localization_energy.yaml"))
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional run directory for virtual_cycle outputs.")
    parser.add_argument("--out", type=Path, default=None, help="Summary JSON output path.")
    parser.add_argument("--min-cycles", type=int, default=3)
    parser.add_argument("--min-crack-delta", type=float, default=5.0e-2)
    parser.add_argument("--min-localization-index", type=float, default=3.0)
    parser.add_argument("--min-energy-crack-mean", type=float, default=1.0e-10)
    parser.add_argument("--min-energy-total-density-mean", type=float, default=1.0e-10)
    parser.add_argument("--max-runtime-warnings", type=int, default=50)
    parser.add_argument("--max-mechanical-not-accepted-steps", type=int, default=160)
    parser.add_argument("--max-crack-cg-nonconverged-steps", type=int, default=20)
    parser.add_argument("--max-nonfinite-count", type=int, default=0)
    parser.add_argument("--min-vtk-energy-fields", type=int, default=4)
    args = parser.parse_args()

    out_path = args.out or _default_out()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir or (out_path.parent / "run")
    run_dir.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")
    vc_raw, _meta = _resolve_payload(raw)
    if not isinstance(vc_raw, dict):
        raise ValueError("virtual_cycle config must be a mapping.")
    cfg = _normalize_config(vc_raw)
    cfg["run_dir"] = run_dir
    cfg["auto_output"] = True
    cfg["export_energy_fields"] = True

    thresholds = Thresholds(
        min_cycles=args.min_cycles,
        min_crack_delta=args.min_crack_delta,
        min_localization_index=args.min_localization_index,
        min_energy_crack_mean=args.min_energy_crack_mean,
        min_energy_total_density_mean=args.min_energy_total_density_mean,
        max_runtime_warnings=args.max_runtime_warnings,
        max_mechanical_not_accepted_steps=args.max_mechanical_not_accepted_steps,
        max_crack_cg_nonconverged_steps=args.max_crack_cg_nonconverged_steps,
        max_nonfinite_count=args.max_nonfinite_count,
        min_vtk_energy_fields=args.min_vtk_energy_fields,
    )

    t0 = datetime.now()
    diagnostics: dict[str, float | int | bool] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        results, paris, coffman = run_virtual_cycles(**cfg, diagnostics_out=diagnostics)
        runtime_warning_count = _runtime_warning_count(caught)
    t1 = datetime.now()

    failures: dict[str, str] = {}
    if not results:
        failures["cycles"] = "no cycles completed"
        first = None
        last = None
    else:
        first = results[0]
        last = results[-1]

    cycles_completed = len(results)
    crack_mean_initial = float(first.crack_mean) if first is not None else 0.0
    crack_mean_final = float(last.crack_mean) if last is not None else 0.0
    crack_delta_total = float(last.crack_length - first.crack_length) if first is not None and last is not None else 0.0
    crack_localization_final = float(last.crack_localization_index) if last is not None else 0.0
    crack_localization_peak = float(max((r.crack_localization_index for r in results), default=0.0))
    energy_crack_mean_final = float(last.energy_crack_mean) if last is not None else 0.0
    energy_total_mean_final = float(last.energy_total_density_mean) if last is not None else 0.0

    mechanical_not_accepted_steps = int(diagnostics.get("mechanical_not_accepted_steps", 0))
    crack_cg_nonconverged_steps = int(diagnostics.get("crack_cg_nonconverged_steps", 0))
    nonfinite_count = int(diagnostics.get("nonfinite_count", 0))

    cycle_csv = run_dir / "virtual_cycle.csv"
    analysis_csv = run_dir / "virtual_cycle_analysis.csv"
    stress_csv = run_dir / "virtual_cycle_stress_strain.csv"
    vtk_dir = run_dir / "vtk"
    vtk_frames = sorted(vtk_dir.glob("anim_frame_*.vtk"))
    vtk_last = vtk_frames[-1] if vtk_frames else None

    required_cycle_cols = {
        "crack_localization_index",
        "energy_elastic_mean",
        "energy_pfc_mean",
        "energy_crack_mean",
        "energy_total_density_mean",
    }
    cycle_header = set(_read_csv_header(cycle_csv))
    missing_cycle_cols = sorted(required_cycle_cols - cycle_header)

    vtk_scalars = _extract_vtk_scalars(vtk_last) if vtk_last is not None else []
    required_vtk_fields = [
        "energy_elastic",
        "energy_pfc",
        "energy_crack",
        "energy_total_density",
        "crack_driving_force",
        "toughness",
    ]
    vtk_energy_fields_present = [name for name in required_vtk_fields if name in vtk_scalars]

    if cycles_completed < thresholds.min_cycles:
        failures["min_cycles"] = f"{cycles_completed} < {thresholds.min_cycles}"
    if crack_delta_total < thresholds.min_crack_delta:
        failures["crack_delta_total"] = f"{crack_delta_total:.6e} < {thresholds.min_crack_delta:.6e}"
    if crack_localization_peak < thresholds.min_localization_index:
        failures["crack_localization_index_peak"] = (
            f"{crack_localization_peak:.6e} < {thresholds.min_localization_index:.6e}"
        )
    if energy_crack_mean_final < thresholds.min_energy_crack_mean:
        failures["energy_crack_mean_final"] = (
            f"{energy_crack_mean_final:.6e} < {thresholds.min_energy_crack_mean:.6e}"
        )
    if energy_total_mean_final < thresholds.min_energy_total_density_mean:
        failures["energy_total_density_mean_final"] = (
            f"{energy_total_mean_final:.6e} < {thresholds.min_energy_total_density_mean:.6e}"
        )
    if runtime_warning_count > thresholds.max_runtime_warnings:
        failures["runtime_warning_count"] = (
            f"{runtime_warning_count} > {thresholds.max_runtime_warnings}"
        )
    if mechanical_not_accepted_steps > thresholds.max_mechanical_not_accepted_steps:
        failures["mechanical_not_accepted_steps"] = (
            f"{mechanical_not_accepted_steps} > {thresholds.max_mechanical_not_accepted_steps}"
        )
    if crack_cg_nonconverged_steps > thresholds.max_crack_cg_nonconverged_steps:
        failures["crack_cg_nonconverged_steps"] = (
            f"{crack_cg_nonconverged_steps} > {thresholds.max_crack_cg_nonconverged_steps}"
        )
    if nonfinite_count > thresholds.max_nonfinite_count:
        failures["nonfinite_count"] = f"{nonfinite_count} > {thresholds.max_nonfinite_count}"
    if missing_cycle_cols:
        failures["cycle_csv_columns"] = f"missing: {','.join(missing_cycle_cols)}"
    if len(vtk_energy_fields_present) < thresholds.min_vtk_energy_fields:
        failures["vtk_energy_fields"] = (
            f"{len(vtk_energy_fields_present)} < {thresholds.min_vtk_energy_fields}"
        )

    metrics = Metrics(
        cycles_completed=cycles_completed,
        crack_mean_initial=crack_mean_initial,
        crack_mean_final=crack_mean_final,
        crack_delta_total=crack_delta_total,
        crack_localization_index_final=crack_localization_final,
        crack_localization_index_peak=crack_localization_peak,
        energy_crack_mean_final=energy_crack_mean_final,
        energy_total_density_mean_final=energy_total_mean_final,
        runtime_warning_count=runtime_warning_count,
        mechanical_not_accepted_steps=mechanical_not_accepted_steps,
        crack_cg_nonconverged_steps=crack_cg_nonconverged_steps,
        nonfinite_count=nonfinite_count,
        vtk_energy_field_count=len(vtk_energy_fields_present),
        vtk_energy_fields_present=vtk_energy_fields_present,
    )

    report = Report(
        passed=len(failures) == 0,
        failures=failures,
        thresholds=thresholds,
        metrics=metrics,
        outputs={
            "run_dir": str(run_dir),
            "cycle_csv": str(cycle_csv),
            "analysis_csv": str(analysis_csv),
            "stress_strain_csv": str(stress_csv),
            "vtk_dir": str(vtk_dir),
            "vtk_last": str(vtk_last) if vtk_last is not None else None,
            "vtk_scalar_fields": vtk_scalars,
        },
        run={
            "config": str(args.config),
            "cycles_completed": cycles_completed,
            "paris_coeff": float(paris) if math.isfinite(float(paris)) else None,
            "coffman_coeff": float(coffman) if math.isfinite(float(coffman)) else None,
            "stability_diagnostics": diagnostics,
        },
        timing={"total_s": (t1 - t0).total_seconds()},
    )

    payload = json.dumps(asdict(report), indent=2)
    out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
