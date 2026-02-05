"""
Microstrain linear-elastic regression check.

Outputs a JSON report to stdout and optionally to a file.
Exit code is non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.energy import CopperParameters, FractureParameters, PFCParameters, PFCCoupling, FreeEnergy
from sim.mechanics import MechanicalConfig, MechanicalEquilibriumSolver
from sim.operators import GridSpec
from sim.pfc import PFCEvolver
from sim.solver import AlternatingSolver, SolverConfig


@dataclass
class Thresholds:
    linear_ratio_tol: float = 0.02
    accum_plastic_max: float = 1e-8


@dataclass
class Results:
    stress_ratio: float
    accum_plastic_mean: float


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


def _test_microstrain() -> Results:
    grid = GridSpec(shape=(12, 8, 4), spacing=(1.0, 1.0, 1.0), periodic=(True, True, False))
    orientation = _make_orientation(grid)
    copper = CopperParameters()
    fracture = FractureParameters()
    pfc_params = PFCParameters()
    coupling = PFCCoupling(pfc_params, fracture, mode="density")
    energy = FreeEnergy(copper, fracture, coupling)
    mech_cfg = MechanicalConfig()
    mechanical = MechanicalEquilibriumSolver(grid, copper, orientation, fracture_k=fracture.k, config=mech_cfg)
    pfc = PFCEvolver(grid, pfc_params, dt=0.0, clip=1.0)
    cfg = SolverConfig(dt=1e-3, plastic_relax=0.2, pfc_active=False)
    solver = AlternatingSolver(coupling, energy, mechanical, pfc, cfg)
    solver.initialize_state(orientation, seed=0)

    eps1 = 1e-6
    eps2 = 2e-6
    nu = 0.34
    macro1 = (eps1, -nu * eps1, -nu * eps1)
    macro2 = (eps2, -nu * eps2, -nu * eps2)
    solver.step(macro1)
    stress1 = solver.state["stress"]
    s1 = float(np.mean(stress1[..., 0, 0]))
    solver.step(macro2)
    stress2 = solver.state["stress"]
    s2 = float(np.mean(stress2[..., 0, 0]))
    ratio = s2 / max(s1, 1e-12)
    accum_plastic = float(np.mean(solver.state.get("accum_plastic", 0.0)))
    return Results(stress_ratio=ratio, accum_plastic_mean=accum_plastic)


def main() -> None:
    parser = argparse.ArgumentParser(description="Microstrain linear-elastic regression.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--linear-tol", type=float, default=0.02, help="Relative tolerance for stress ratio.")
    parser.add_argument("--plastic-max", type=float, default=1e-8, help="Max mean accumulated plastic.")
    args = parser.parse_args()

    thresholds = Thresholds(linear_ratio_tol=args.linear_tol, accum_plastic_max=args.plastic_max)
    failures: Dict[str, str] = {}
    t0 = perf_counter()
    results = _test_microstrain()
    t1 = perf_counter()

    if abs(results.stress_ratio - 2.0) / 2.0 > thresholds.linear_ratio_tol:
        failures["stress_ratio"] = f"ratio {results.stress_ratio:.6e} exceeds tolerance"
    if results.accum_plastic_mean > thresholds.accum_plastic_max:
        failures["accum_plastic"] = f"{results.accum_plastic_mean:.6e} exceeds max"

    report = Report(
        results=results,
        thresholds=thresholds,
        timing={"total_s": t1 - t0},
        passed=len(failures) == 0,
        failures=failures,
    )
    payload = json.dumps(asdict(report), indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
