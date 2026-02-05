"""
Micron-scale regression checks (spacing set to 1e-6).

Outputs a JSON report to stdout and optionally to a file.
Exit code is non-zero on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Dict, Tuple

import numpy as np

from sim.energy import CopperParameters, FractureParameters, PFCParameters, PFCCoupling, FreeEnergy
from sim.mechanics import MechanicalEquilibriumSolver, divergence
from sim.operators import GridSpec
from sim.pfc import PFCEvolver
from sim.solver import AlternatingSolver, SolverConfig


@dataclass
class Thresholds:
    compression_phi_growth_max: float = 1e-8
    patch_div_norm_max: float = 1e-8
    patch_stress_std_ratio_max: float = 1e-8
    modeI_phi_mean_delta_min: float = 1e-6


@dataclass
class Results:
    compression_phi_growth: float
    patch_div_norm: float
    patch_stress_std_ratio: float
    modeI_phi_mean_start: float
    modeI_phi_mean_end: float
    modeI_phi_mean_delta: float


@dataclass
class Report:
    results: Results
    thresholds: Thresholds
    timing: Dict[str, float]
    passed: bool
    failures: Dict[str, str]


def _make_orientation(grid: GridSpec) -> np.ndarray:
    eye = np.eye(3)
    return np.broadcast_to(eye, grid.shape + (3, 3)).copy()


def _test_pure_compression(spacing: Tuple[float, float, float]) -> float:
    grid = GridSpec(shape=(16, 8, 4), spacing=spacing, periodic=(True, True, False))
    orientation = _make_orientation(grid)
    copper = CopperParameters()
    scale = spacing[0]
    fracture = FractureParameters(gc=scale, l0=scale)
    pfc_params = PFCParameters()
    coupling = PFCCoupling(pfc_params, fracture, mode="density")
    energy = FreeEnergy(copper, fracture, coupling)
    mechanical = MechanicalEquilibriumSolver(grid, copper, orientation, fracture_k=fracture.k)
    pfc = PFCEvolver(grid, pfc_params, dt=0.0, clip=1.0)
    cfg = SolverConfig(dt=1e-2, crack_relax=1.0, plastic_relax=0.1, mech_plastic_weight=0.8)
    solver = AlternatingSolver(coupling, energy, mechanical, pfc, cfg)
    solver.initialize_state(orientation, seed=0)

    crack = solver.state["crack"]
    crack[6:9, 3:5, 1:3] = 0.6
    solver.state["crack"] = crack

    phi0 = solver.state["crack"].copy()
    for _ in range(5):
        solver.step((-0.005, -0.005, -0.005))
    phi1 = solver.state["crack"]
    return float(np.max(phi1 - phi0))


def _test_patch(spacing: Tuple[float, float, float]) -> Tuple[float, float]:
    grid = GridSpec(shape=(12, 12, 12), spacing=spacing, periodic=(True, True, True))
    orientation = _make_orientation(grid)
    copper = CopperParameters()
    mechanical = MechanicalEquilibriumSolver(grid, copper, orientation)
    disp = np.zeros(grid.shape + (3,))
    crack = np.zeros(grid.shape)
    macro = (0.002, -0.0005, -0.0005)
    _, _, stress = mechanical.solve(disp, crack, macro)
    div = divergence(stress, grid.spacing, grid.periodic)
    div_norm = float(np.linalg.norm(div))
    stress_mean = np.mean(stress, axis=(0, 1, 2))
    stress_std = np.std(stress, axis=(0, 1, 2))
    ratio = float(np.linalg.norm(stress_std) / (np.linalg.norm(stress_mean) + 1e-12))
    return div_norm, ratio


def _test_mode_I(spacing: Tuple[float, float, float]) -> Tuple[float, float]:
    grid = GridSpec(shape=(24, 12, 4), spacing=spacing, periodic=(True, True, False))
    orientation = _make_orientation(grid)
    copper = CopperParameters()
    scale = spacing[0]
    fracture = FractureParameters(gc=scale, l0=scale, k=1e-6, epsilon_half=0.15, gres=0.1)
    pfc_params = PFCParameters()
    coupling = PFCCoupling(pfc_params, fracture, mode="density")
    energy = FreeEnergy(copper, fracture, coupling)
    mechanical = MechanicalEquilibriumSolver(grid, copper, orientation, fracture_k=fracture.k)
    pfc = PFCEvolver(grid, pfc_params, dt=0.0, clip=1.0)
    cfg = SolverConfig(dt=1e-2, crack_relax=1.0, plastic_relax=0.1, mech_plastic_weight=0.8)
    solver = AlternatingSolver(coupling, energy, mechanical, pfc, cfg)
    solver.initialize_state(orientation, seed=0)

    crack = solver.state["crack"]
    crack[3:6, 5:7, 1:3] = 0.7
    solver.state["crack"] = crack

    phi0 = float(np.mean(crack))
    for _ in range(8):
        solver.step((0.006, 0.0, 0.0))
    phi1 = float(np.mean(solver.state["crack"]))
    return phi0, phi1


def run(thr: Thresholds) -> Report:
    spacing = (1e-6, 1e-6, 1e-6)
    timings: Dict[str, float] = {}
    t0 = perf_counter()
    t = perf_counter()
    comp_growth = _test_pure_compression(spacing)
    timings["compression_s"] = perf_counter() - t
    t = perf_counter()
    div_norm, ratio = _test_patch(spacing)
    timings["patch_s"] = perf_counter() - t
    t = perf_counter()
    phi0, phi1 = _test_mode_I(spacing)
    timings["modeI_s"] = perf_counter() - t
    timings["total_s"] = perf_counter() - t0
    delta = phi1 - phi0

    failures: Dict[str, str] = {}
    if comp_growth > thr.compression_phi_growth_max:
        failures["compression"] = f"phi_growth {comp_growth:.3e} > {thr.compression_phi_growth_max:.3e}"
    if div_norm > thr.patch_div_norm_max:
        failures["patch_div"] = f"div_norm {div_norm:.3e} > {thr.patch_div_norm_max:.3e}"
    if ratio > thr.patch_stress_std_ratio_max:
        failures["patch_stress"] = f"stress_std_ratio {ratio:.3e} > {thr.patch_stress_std_ratio_max:.3e}"
    if delta < thr.modeI_phi_mean_delta_min:
        failures["modeI"] = f"phi_mean_delta {delta:.3e} < {thr.modeI_phi_mean_delta_min:.3e}"

    report = Report(
        results=Results(
            compression_phi_growth=comp_growth,
            patch_div_norm=div_norm,
            patch_stress_std_ratio=ratio,
            modeI_phi_mean_start=phi0,
            modeI_phi_mean_end=phi1,
            modeI_phi_mean_delta=delta,
        ),
        thresholds=thr,
        timing=timings,
        passed=len(failures) == 0,
        failures=failures,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="", help="Optional JSON output path (default: stdout only)")
    parser.add_argument("--strict", action="store_true", help="Use stricter thresholds")
    parser.add_argument("--compression-max", type=float, default=Thresholds.compression_phi_growth_max)
    parser.add_argument("--patch-div-max", type=float, default=Thresholds.patch_div_norm_max)
    parser.add_argument("--patch-std-ratio-max", type=float, default=Thresholds.patch_stress_std_ratio_max)
    parser.add_argument("--modeI-delta-min", type=float, default=Thresholds.modeI_phi_mean_delta_min)
    args = parser.parse_args()

    if args.strict:
        thr = Thresholds(
            compression_phi_growth_max=min(args.compression_max, 1e-10),
            patch_div_norm_max=min(args.patch_div_max, 1e-10),
            patch_stress_std_ratio_max=min(args.patch_std_ratio_max, 1e-10),
            modeI_phi_mean_delta_min=max(args.modeI_delta_min, 1e-5),
        )
    else:
        thr = Thresholds(
            compression_phi_growth_max=args.compression_max,
            patch_div_norm_max=args.patch_div_max,
            patch_stress_std_ratio_max=args.patch_std_ratio_max,
            modeI_phi_mean_delta_min=args.modeI_delta_min,
        )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "script": "regress_bc_crack_micron",
            "args": vars(args),
            "thresholds": asdict(thr),
        }
        (output_path.parent / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    report = run(thr)

    payload = {
        "passed": report.passed,
        "results": asdict(report.results),
        "thresholds": asdict(report.thresholds),
        "timing": report.timing,
        "failures": report.failures,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
