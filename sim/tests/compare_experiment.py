"""
Compare experimental stress-strain data against simulation outputs.

Features:
- Load a specific cycle file (200 samples) from the experimental folder.
- Apply Schmid factor correction: tau = m * stress, gamma = strain / m.
- Optionally load simulation CSV (virtual_cycle_stress_strain.csv) and apply the same correction.
- Export corrected CSV and overlay plot (if matplotlib is available).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SchmidResult:
    m: float
    axis: np.ndarray


def _parse_vec(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 components, got {parts}")
    vec = np.array([float(p) for p in parts], dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Vector must be non-zero")
    return vec / norm


def schmid_factor(axis: np.ndarray, plane: np.ndarray, direction: np.ndarray) -> float:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    plane = plane / (np.linalg.norm(plane) + 1e-12)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return float(abs(np.dot(axis, plane) * np.dot(axis, direction)))


def max_schmid_fcc(axis: np.ndarray) -> SchmidResult:
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
                m = abs(np.dot(axis, n) * np.dot(axis, d))
                if m > m_max:
                    m_max = m
    return SchmidResult(m=m_max, axis=axis)


def _default_run_dir(task: str) -> Path:
    date_str = date.today().isoformat()
    time_str = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/runs") / date_str / f"{task}_{time_str}"


def _resolve_cycle_file(folder: Path, cycle: int) -> Path:
    pattern = f"*data_{cycle:08d}.txt"
    matches = list(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No cycle file found for pattern {pattern} in {folder}")
    if len(matches) > 1:
        matches.sort()
    return matches[0]


def _load_experiment(
    path: Path,
    stress_col: int,
    strain_col: int,
    stress_scale: float = 1.0,
    strain_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] <= max(stress_col, strain_col):
        raise ValueError(f"Unexpected data shape {data.shape} in {path}")
    stress = data[:, stress_col] * stress_scale
    strain = data[:, strain_col] * strain_scale
    return stress, strain


def _maybe_load_sim(path: Path, stress_col: str, strain_col: str, stress_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    import csv

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Simulation CSV empty: {path}")
    if stress_col not in rows[0] or strain_col not in rows[0]:
        raise ValueError(f"Simulation CSV missing columns: {stress_col} / {strain_col}")
    stress = np.array([float(r[stress_col]) for r in rows], dtype=float) * stress_scale
    strain = np.array([float(r[strain_col]) for r in rows], dtype=float)
    return stress, strain


def _write_csv(path: Path, stress: np.ndarray, strain: np.ndarray, tau: np.ndarray, gamma: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("stress_raw,strain_raw,tau_schmid,gamma_schmid\n")
        for s, e, t, g in zip(stress, strain, tau, gamma):
            fh.write(f"{s:.6e},{e:.6e},{t:.6e},{g:.6e}\n")


def _plot_overlay(out_path: Path, exp_gamma: np.ndarray, exp_tau: np.ndarray, sim_gamma: np.ndarray | None, sim_tau: np.ndarray | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib not available, skip plot")
        return
    plt.figure(figsize=(6, 4))
    plt.plot(exp_gamma, exp_tau, "-", lw=1.5, label="experiment (schmid)")
    if sim_gamma is not None and sim_tau is not None:
        plt.plot(sim_gamma, sim_tau, "--", lw=1.2, label="simulation (schmid)")
    plt.xlabel("gamma (shear strain)")
    plt.ylabel("tau (shear stress)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare experimental cycle data with simulation.")
    parser.add_argument("--exp-folder", type=Path, required=True, help="Folder containing *_data_XXXXXXXX.txt files.")
    parser.add_argument("--cycle", type=int, default=1000, help="Cycle index to load (0-based file index).")
    parser.add_argument("--stress-col", type=int, default=0, help="Column index for stress (0-based).")
    parser.add_argument("--strain-col", type=int, default=1, help="Column index for strain (0-based).")
    parser.add_argument("--exp-stress-scale", type=float, default=1.0, help="Scale experimental stress.")
    parser.add_argument("--exp-strain-scale", type=float, default=1.0, help="Scale experimental strain.")
    parser.add_argument("--axis", type=str, default="-1,-1,-1", help="Load axis (e.g., -1,-1,-1).")
    parser.add_argument("--plane", type=str, default=None, help="Slip plane normal (e.g., 1,1,1).")
    parser.add_argument("--direction", type=str, default=None, help="Slip direction (e.g., -1,0,1).")
    parser.add_argument("--use-max", action="store_true", help="Use max Schmid factor across {111}<110>.")
    parser.add_argument("--schmid", type=float, default=None, help="Override Schmid factor.")
    parser.add_argument("--sim-csv", type=Path, default=None, help="Simulation stress-strain CSV to compare.")
    parser.add_argument("--sim-stress-col", type=str, default="sig_xx_GPa", help="Sim CSV stress column name.")
    parser.add_argument("--sim-strain-col", type=str, default="macro_strain", help="Sim CSV strain column name.")
    parser.add_argument("--sim-stress-scale", type=float, default=1000.0, help="Scale sim stress (GPa->MPa=1000).")
    parser.add_argument("--run-dir", type=Path, default=None, help="Output directory.")
    args = parser.parse_args()

    axis = _parse_vec(args.axis)
    if args.schmid is not None:
        schmid = float(args.schmid)
    elif args.use_max:
        schmid = max_schmid_fcc(axis).m
    else:
        if args.plane is None or args.direction is None:
            raise ValueError("Provide --plane and --direction, or use --use-max/--schmid")
        plane = _parse_vec(args.plane)
        direction = _parse_vec(args.direction)
        schmid = schmid_factor(axis, plane, direction)

    if schmid <= 0:
        raise ValueError(f"Schmid factor is {schmid:.6e}, cannot apply correction.")

    cycle_file = _resolve_cycle_file(args.exp_folder, args.cycle)
    stress_raw, strain_raw = _load_experiment(
        cycle_file,
        args.stress_col,
        args.strain_col,
        stress_scale=args.exp_stress_scale,
        strain_scale=args.exp_strain_scale,
    )
    tau = schmid * stress_raw
    gamma = strain_raw / schmid

    run_dir = args.run_dir or _default_run_dir(f"exp_compare_cycle_{args.cycle:08d}")
    out_csv = run_dir / "experiment_schmid.csv"
    _write_csv(out_csv, stress_raw, strain_raw, tau, gamma)

    sim_tau = sim_gamma = None
    if args.sim_csv is not None:
        sim_stress, sim_strain = _maybe_load_sim(args.sim_csv, args.sim_stress_col, args.sim_strain_col, args.sim_stress_scale)
        sim_tau = schmid * sim_stress
        sim_gamma = sim_strain / schmid
        _write_csv(run_dir / "simulation_schmid.csv", sim_stress, sim_strain, sim_tau, sim_gamma)

    _plot_overlay(run_dir / "schmid_overlay.png", gamma, tau, sim_gamma, sim_tau)
    print(f"[ok] Schmid={schmid:.6f} | exp={cycle_file.name}")
    print(f"[ok] Output: {out_csv}")
    if args.sim_csv is not None:
        print(f"[ok] Sim: {args.sim_csv}")


if __name__ == "__main__":
    main()
