"""
Larger-grid, multi-cycle regression checks for boundary handling and crack driving.

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
    modeI_phi_mean_delta_min: float = 1e-5
    modeI_compression_growth_max: float = 1e-5


@dataclass
class Results:
    compression_phi_growth: float
    patch_div_norm: float
    patch_stress_std_ratio: float
    modeI_phi_mean_start: float
    modeI_phi_mean_end: float
    modeI_phi_mean_delta: float
    modeI_compression_growth_max: float


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


def _compression_test(
    grid_shape: Tuple[int, int, int],
    steps: int,
    eps: float,
) -> float:
    grid = GridSpec(shape=grid_shape, spacing=(1.0, 1.0, 1.0), periodic=(True, True, False))
    orientation = _make_orientation(grid)
    copper = CopperParameters()
    fracture = FractureParameters()
    pfc_params = PFCParameters()
    coupling = PFCCoupling(pfc_params, fracture, mode="density")
    energy = FreeEnergy(copper, fracture, coupling)
    mechanical = MechanicalEquilibriumSolver(grid, copper, orientation, fracture_k=fracture.k)
    pfc = PFCEvolver(grid, pfc_params, dt=0.0, clip=1.0)
    cfg = SolverConfig(dt=1e-2, crack_relax=1.0, plastic_relax=0.1, mech_plastic_weight=0.8)
    solver = AlternatingSolver(coupling, energy, mechanical, pfc, cfg)
    solver.initialize_state(orientation, seed=0)

    crack = solver.state["crack"]
    crack[8:12, 4:6, 2:4] = 0.6
    solver.state["crack"] = crack

    phi0 = solver.state["crack"].copy()
    for _ in range(steps):
        # tri-axial compression to avoid accidental tensile directions
        solver.step((eps, eps, eps))
    phi1 = solver.state["crack"]
    return float(np.max(phi1 - phi0))


def _patch_test(grid_shape: Tuple[int, int, int]) -> Tuple[float, float]:
    grid = GridSpec(shape=grid_shape, spacing=(1.0, 1.0, 1.0), periodic=(True, True, True))
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


def _modeI_cyclic(
    grid_shape: Tuple[int, int, int],
    cycles: int,
    seg_steps: int,
    eps_max: float,
    eps_min: float,
) -> Tuple[float, float, float]:
    grid = GridSpec(shape=grid_shape, spacing=(1.0, 1.0, 1.0), periodic=(True, True, False))
    orientation = _make_orientation(grid)
    copper = CopperParameters()
    fracture = FractureParameters(gc=0.6, l0=1.0, k=1e-6, epsilon_half=0.15, gres=0.1)
    pfc_params = PFCParameters()
    coupling = PFCCoupling(pfc_params, fracture, mode="density")
    energy = FreeEnergy(copper, fracture, coupling)
    mechanical = MechanicalEquilibriumSolver(grid, copper, orientation, fracture_k=fracture.k)
    pfc = PFCEvolver(grid, pfc_params, dt=0.0, clip=1.0)
    cfg = SolverConfig(dt=1e-2, crack_relax=1.0, plastic_relax=0.1, mech_plastic_weight=0.8)
    solver = AlternatingSolver(coupling, energy, mechanical, pfc, cfg)
    solver.initialize_state(orientation, seed=0)

    crack = solver.state["crack"]
    crack[5:8, 7:9, 2:5] = 0.7
    solver.state["crack"] = crack

    phi_mean_start = float(np.mean(crack))
    phi_prev = phi_mean_start
    comp_growth_max = 0.0

    segments = [eps_max, 0.0, eps_min, 0.0]
    current = 0.0

    for _ in range(cycles):
        for target in segments:
            start = current
            for step in range(1, seg_steps + 1):
                alpha = step / seg_steps
                current = start + (target - start) * alpha
                macro = (current, 0.0, 0.0)
                solver.step(macro)
                phi_now = float(np.mean(solver.state["crack"]))
                if current < 0.0:
                    comp_growth_max = max(comp_growth_max, phi_now - phi_prev)
                phi_prev = phi_now

    phi_mean_end = float(np.mean(solver.state["crack"]))
    return phi_mean_start, phi_mean_end, comp_growth_max


def run(thr: Thresholds) -> Report:
    timings: Dict[str, float] = {}
    t0 = perf_counter()
    t = perf_counter()
    comp_growth = _compression_test(grid_shape=(24, 12, 6), steps=5, eps=-0.005)
    timings["compression_s"] = perf_counter() - t
    t = perf_counter()
    div_norm, ratio = _patch_test(grid_shape=(20, 20, 20))
    timings["patch_s"] = perf_counter() - t
    t = perf_counter()
    phi0, phi1, comp_max = _modeI_cyclic(
        grid_shape=(32, 16, 8), cycles=2, seg_steps=5, eps_max=0.006, eps_min=-0.003
    )
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
        failures["modeI_delta"] = f"phi_mean_delta {delta:.3e} < {thr.modeI_phi_mean_delta_min:.3e}"
    if comp_max > thr.modeI_compression_growth_max:
        failures["modeI_comp"] = f"compression_phi_growth {comp_max:.3e} > {thr.modeI_compression_growth_max:.3e}"

    report = Report(
        results=Results(
            compression_phi_growth=comp_growth,
            patch_div_norm=div_norm,
            patch_stress_std_ratio=ratio,
            modeI_phi_mean_start=phi0,
            modeI_phi_mean_end=phi1,
            modeI_phi_mean_delta=delta,
            modeI_compression_growth_max=comp_max,
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
    parser.add_argument("--modeI-comp-max", type=float, default=Thresholds.modeI_compression_growth_max)
    args = parser.parse_args()

    if args.strict:
        thr = Thresholds(
            compression_phi_growth_max=min(args.compression_max, 1e-9),
            patch_div_norm_max=min(args.patch_div_max, 1e-9),
            patch_stress_std_ratio_max=min(args.patch_std_ratio_max, 1e-9),
            modeI_phi_mean_delta_min=max(args.modeI_delta_min, 5e-5),
            modeI_compression_growth_max=min(args.modeI_comp_max, 1e-6),
        )
    else:
        thr = Thresholds(
            compression_phi_growth_max=args.compression_max,
            patch_div_norm_max=args.patch_div_max,
            patch_stress_std_ratio_max=args.patch_std_ratio_max,
            modeI_phi_mean_delta_min=args.modeI_delta_min,
            modeI_compression_growth_max=args.modeI_comp_max,
        )
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
