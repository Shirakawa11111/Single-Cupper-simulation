"""
Experiment-alignment regression gate for low-amplitude fatigue trajectory.

Default workflow:
1) Run virtual-cycle simulation from a YAML config.
2) Compare Schmid-corrected simulation and experiment curves.
3) Fail on metric/stability threshold violations.
"""

from __future__ import annotations

import argparse
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
    rmse_tau_max: float = 30.0
    mae_tau_max: float = 25.0
    rmse_gamma_max: float = 4.2e-3
    max_runtime_warnings: int = 50
    max_mechanical_not_accepted_steps: int = 160
    max_crack_cg_nonconverged_steps: int = 40
    max_nonfinite_count: int = 0


@dataclass
class Metrics:
    rmse_tau_MPa: float
    mae_tau_MPa: float
    rmse_gamma: float
    mae_gamma: float
    tau_range_MPa: float
    gamma_range: float
    n_exp: int
    n_sim: int


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
    return Path("sim/tests/regress_runs") / day / f"exp_alignment_gate_{ts}" / "summary.json"


def _default_run_dir() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"exp_alignment_gate_run_{ts}"


def _parse_vec(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 components, got: {parts}")
    vec = np.array([float(p) for p in parts], dtype=float)
    nrm = np.linalg.norm(vec)
    if nrm < 1e-12:
        raise ValueError("Axis vector must be non-zero")
    return vec / nrm


def _max_schmid_fcc(axis: np.ndarray) -> float:
    normals = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    dirs = np.array(
        [
            [0, 1, -1],
            [0, 1, 1],
            [1, 0, -1],
            [1, 0, 1],
            [1, -1, 0],
            [1, 1, 0],
        ],
        dtype=float,
    )
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    m_max = 0.0
    for n in normals:
        for d in dirs:
            if abs(np.dot(n, d)) < 1e-6:
                m_max = max(m_max, abs(np.dot(axis, n) * np.dot(axis, d)))
    return float(m_max)


def _resolve_cycle_file(folder: Path, cycle: int) -> Path:
    pattern = f"*data_{cycle:08d}.txt"
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No experiment file matches {pattern} in {folder}")
    return matches[0]


def _load_experiment(path: Path, stress_col: int, strain_col: int, strain_scale: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected experiment shape: {arr.shape}")
    if arr.shape[1] <= max(stress_col, strain_col):
        raise ValueError(f"Experiment has {arr.shape[1]} columns, requested {stress_col}/{strain_col}")
    stress = arr[:, stress_col]
    strain = arr[:, strain_col] * strain_scale
    return stress.astype(float), strain.astype(float)


def _load_sim_csv(path: Path, stress_col: str, strain_col: str, stress_scale: float) -> tuple[np.ndarray, np.ndarray]:
    import csv

    with path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"Simulation CSV is empty: {path}")
    if stress_col not in rows[0] or strain_col not in rows[0]:
        raise ValueError(f"Simulation CSV missing {stress_col}/{strain_col}")
    stress = np.array([float(r[stress_col]) for r in rows], dtype=float) * stress_scale
    strain = np.array([float(r[strain_col]) for r in rows], dtype=float)
    return stress, strain


def _interp_phase(values: np.ndarray, target_len: int) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, values)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _runtime_warning_count(caught: list[warnings.WarningMessage]) -> int:
    total = 0
    for w in caught:
        if issubclass(w.category, RuntimeWarning):
            total += 1
    return total


def _prepare_runtime_cfg(config_path: Path, run_dir: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    vc_raw, _meta = _resolve_payload(raw)
    cfg = _normalize_config(vc_raw)

    # Force dynamic output dir for regression runs.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment-alignment regression gate.")
    parser.add_argument("--config", type=Path, default=Path("sim/configs/fatigue_lowamp_align_locked_v3.yaml"))
    parser.add_argument("--sim-csv", type=Path, default=None, help="Use existing simulation CSV and skip simulation run.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory when simulation is executed.")
    parser.add_argument("--out", type=Path, default=None, help="Output summary JSON path.")
    parser.add_argument("--exp-folder", type=Path, default=Path("20240917_-111_Cu_1e-3_4000_294K_001"))
    parser.add_argument("--cycle", type=int, default=1000)
    parser.add_argument("--stress-col", type=int, default=1)
    parser.add_argument("--strain-col", type=int, default=0)
    parser.add_argument("--exp-strain-scale", type=float, default=1e-6)
    parser.add_argument("--axis", type=str, default="-1,1,1")
    parser.add_argument("--sim-stress-col", type=str, default="sig_xx_GPa")
    parser.add_argument("--sim-strain-col", type=str, default="macro_strain")
    parser.add_argument("--sim-stress-scale", type=float, default=1000.0)
    parser.add_argument("--rmse-tau-max", type=float, default=30.0)
    parser.add_argument("--mae-tau-max", type=float, default=25.0)
    parser.add_argument("--rmse-gamma-max", type=float, default=4.2e-3)
    parser.add_argument("--max-runtime-warnings", type=int, default=50)
    parser.add_argument("--max-mechanical-not-accepted-steps", type=int, default=160)
    parser.add_argument("--max-crack-cg-nonconverged-steps", type=int, default=40)
    parser.add_argument("--max-nonfinite-count", type=int, default=0)
    args = parser.parse_args()

    out_path = args.out or _default_out()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    thresholds = Thresholds(
        rmse_tau_max=args.rmse_tau_max,
        mae_tau_max=args.mae_tau_max,
        rmse_gamma_max=args.rmse_gamma_max,
        max_runtime_warnings=args.max_runtime_warnings,
        max_mechanical_not_accepted_steps=args.max_mechanical_not_accepted_steps,
        max_crack_cg_nonconverged_steps=args.max_crack_cg_nonconverged_steps,
        max_nonfinite_count=args.max_nonfinite_count,
    )

    t0 = perf_counter()
    run_info: dict[str, Any] = {}
    diag: dict[str, Any] = {}
    runtime_warning_count = 0

    sim_csv: Path
    if args.sim_csv is not None:
        sim_csv = args.sim_csv
        run_info["mode"] = "external_sim_csv"
        run_info["sim_csv"] = str(sim_csv)
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

        sim_csv = run_dir / "virtual_cycle_stress_strain.csv"
        run_info.update(
            {
                "mode": "run_from_config",
                "config": str(args.config),
                "run_dir": str(run_dir),
                "sim_csv": str(sim_csv),
                "cycles_completed": len(results),
                "paris_coeff": float(paris) if math.isfinite(float(paris)) else None,
                "coffman_coeff": float(coffman) if math.isfinite(float(coffman)) else None,
                "runtime_warning_count": runtime_warning_count,
                "stability_diagnostics": diag,
            }
        )

    axis = _parse_vec(args.axis)
    schmid = _max_schmid_fcc(axis)
    if schmid <= 0:
        raise ValueError(f"Invalid Schmid factor: {schmid}")

    exp_file = _resolve_cycle_file(args.exp_folder, args.cycle)
    exp_stress, exp_strain = _load_experiment(exp_file, args.stress_col, args.strain_col, args.exp_strain_scale)
    sim_stress, sim_strain = _load_sim_csv(sim_csv, args.sim_stress_col, args.sim_strain_col, args.sim_stress_scale)

    exp_tau = exp_stress * schmid
    exp_gamma = exp_strain / schmid
    sim_tau = sim_stress * schmid
    sim_gamma = sim_strain / schmid

    sim_tau_i = _interp_phase(sim_tau, len(exp_tau))
    sim_gamma_i = _interp_phase(sim_gamma, len(exp_gamma))

    metrics = Metrics(
        rmse_tau_MPa=_rmse(sim_tau_i, exp_tau),
        mae_tau_MPa=_mae(sim_tau_i, exp_tau),
        rmse_gamma=_rmse(sim_gamma_i, exp_gamma),
        mae_gamma=_mae(sim_gamma_i, exp_gamma),
        tau_range_MPa=float(np.max(sim_tau) - np.min(sim_tau)),
        gamma_range=float(np.max(sim_gamma) - np.min(sim_gamma)),
        n_exp=int(len(exp_tau)),
        n_sim=int(len(sim_tau)),
    )

    failures: dict[str, str] = {}
    if metrics.rmse_tau_MPa > thresholds.rmse_tau_max:
        failures["rmse_tau_MPa"] = f"{metrics.rmse_tau_MPa:.6f} > {thresholds.rmse_tau_max:.6f}"
    if metrics.mae_tau_MPa > thresholds.mae_tau_max:
        failures["mae_tau_MPa"] = f"{metrics.mae_tau_MPa:.6f} > {thresholds.mae_tau_max:.6f}"
    if metrics.rmse_gamma > thresholds.rmse_gamma_max:
        failures["rmse_gamma"] = f"{metrics.rmse_gamma:.6e} > {thresholds.rmse_gamma_max:.6e}"

    if args.sim_csv is None:
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

    run_info["experiment_file"] = str(exp_file)
    run_info["schmid"] = schmid

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
