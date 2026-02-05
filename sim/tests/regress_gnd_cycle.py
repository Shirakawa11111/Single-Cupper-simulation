"""
Low-amplitude cyclic GND growth regression.

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
from typing import Dict, List, Tuple

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
    gnd_growth_min: float = 1e-5
    accum_growth_min: float = 5e-5
    slope_min: float = 0.0


@dataclass
class Results:
    gnd_means: List[float]
    gnd_max: List[float]
    accum_means: List[float]
    gnd_growth: float
    accum_growth: float
    gnd_slope: float


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


def _run_cycles(
    cycles: int,
    max_strain: float,
    segment_steps: int,
    dt: float,
    gamma0: float,
    slip_exponent: float,
    h_iso: float,
    h_gnd: float,
    plastic_relax: float,
    pfc_active: bool,
) -> Tuple[List[float], List[float], List[float]]:
    grid = GridSpec(shape=(16, 8, 4), spacing=(1.0, 1.0, 1.0), periodic=(True, True, False))
    orientation = _make_orientation(grid)
    copper = CopperParameters()
    fracture = FractureParameters()
    pfc_params = PFCParameters()
    coupling = PFCCoupling(
        pfc_params,
        fracture,
        mode="density",
        gamma0=gamma0,
        slip_exponent=slip_exponent,
        h_iso=h_iso,
        h_gnd=h_gnd,
    )
    energy = FreeEnergy(copper, fracture, coupling)
    mech_cfg = MechanicalConfig()
    mechanical = MechanicalEquilibriumSolver(grid, copper, orientation, fracture_k=fracture.k, config=mech_cfg)
    pfc = PFCEvolver(grid, pfc_params, dt=0.0, clip=1.0)
    cfg = SolverConfig(
        dt=dt,
        plastic_relax=plastic_relax,
        crack_relax=0.0,
        mech_plastic_weight=1.0,
        pfc_active=pfc_active,
        gnd_active=True,
    )
    solver = AlternatingSolver(coupling, energy, mechanical, pfc, cfg)
    solver.initialize_state(orientation, seed=0)
    crack = solver.state["crack"]
    crack[6:9, 3:5, 1:3] = 0.6
    solver.state["crack"] = crack

    min_strain = -max_strain
    load_segments = [max_strain, 0.0, min_strain, 0.0]
    current_strain = 0.0
    gnd_means: List[float] = []
    gnd_max: List[float] = []
    accum_means: List[float] = []

    for _ in range(1, cycles + 1):
        for target in load_segments:
            target_start = current_strain
            target_end = target
            for step in range(1, segment_steps + 1):
                alpha = step / segment_steps
                current_strain = target_start + (target_end - target_start) * alpha
                macro = (current_strain, -0.34 * current_strain, -0.34 * current_strain)
                solver.step(macro)
        gnd = solver.state.get("gnd_density")
        gnd_means.append(float(np.mean(gnd)) if gnd is not None else 0.0)
        gnd_max.append(float(np.max(gnd)) if gnd is not None else 0.0)
        accum_means.append(float(np.mean(solver.state.get("accum_plastic", 0.0))))

    return gnd_means, gnd_max, accum_means


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-amplitude cyclic GND growth regression.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--cycles", type=int, default=5, help="Number of cycles.")
    parser.add_argument("--max-strain", type=float, default=5e-3, help="Max macro strain amplitude.")
    parser.add_argument("--segment-steps", type=int, default=20, help="Steps per segment.")
    parser.add_argument("--dt", type=float, default=5e-3, help="Time step.")
    parser.add_argument("--gamma0", type=float, default=1e-2, help="Slip rate scale.")
    parser.add_argument("--slip-exponent", type=float, default=8.0, help="Slip exponent n.")
    parser.add_argument("--h-iso", type=float, default=4e-4, help="Isotropic hardening.")
    parser.add_argument("--h-gnd", type=float, default=0.0, help="GND hardening.")
    parser.add_argument("--plastic-relax", type=float, default=0.2, help="Plastic relax factor.")
    parser.add_argument("--pfc-active", action="store_true", help="Enable PFC evolution.")
    parser.add_argument("--gnd-growth-min", type=float, default=1e-5, help="Min GND mean growth.")
    parser.add_argument("--accum-growth-min", type=float, default=5e-5, help="Min accum_plastic growth.")
    parser.add_argument("--slope-min", type=float, default=0.0, help="Min slope for GND trend.")
    args = parser.parse_args()

    thresholds = Thresholds(
        gnd_growth_min=args.gnd_growth_min,
        accum_growth_min=args.accum_growth_min,
        slope_min=args.slope_min,
    )
    failures: Dict[str, str] = {}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args_dict = {
            key: (str(val) if isinstance(val, Path) else val)
            for key, val in vars(args).items()
        }
        params = {
            "script": "regress_gnd_cycle",
            "args": args_dict,
            "grid": {"shape": [16, 8, 4], "spacing": [1.0, 1.0, 1.0], "periodic": [True, True, False]},
        }
        params_path = args.output.parent / "params.json"
        params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    t0 = perf_counter()
    gnd_means, gnd_max, accum_means = _run_cycles(
        cycles=args.cycles,
        max_strain=args.max_strain,
        segment_steps=args.segment_steps,
        dt=args.dt,
        gamma0=args.gamma0,
        slip_exponent=args.slip_exponent,
        h_iso=args.h_iso,
        h_gnd=args.h_gnd,
        plastic_relax=args.plastic_relax,
        pfc_active=args.pfc_active,
    )
    t1 = perf_counter()

    cycles_arr = np.arange(1, len(gnd_means) + 1, dtype=float)
    gnd_growth = gnd_means[-1] - gnd_means[0]
    accum_growth = accum_means[-1] - accum_means[0]
    slope = float(np.polyfit(cycles_arr, np.array(gnd_means), 1)[0]) if len(gnd_means) > 1 else 0.0

    if gnd_growth < thresholds.gnd_growth_min:
        failures["gnd_growth"] = f"{gnd_growth:.6e} below min"
    if accum_growth < thresholds.accum_growth_min:
        failures["accum_growth"] = f"{accum_growth:.6e} below min"
    if slope < thresholds.slope_min:
        failures["gnd_slope"] = f"{slope:.6e} below min"

    report = Report(
        results=Results(
            gnd_means=gnd_means,
            gnd_max=gnd_max,
            accum_means=accum_means,
            gnd_growth=gnd_growth,
            accum_growth=accum_growth,
            gnd_slope=slope,
        ),
        thresholds=thresholds,
        timing={"total_s": t1 - t0},
        passed=len(failures) == 0,
        failures=failures,
    )
    payload = json.dumps(asdict(report), indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        csv_path = args.output.parent / "gnd_cycle.csv"
        with csv_path.open("w", encoding="utf-8") as fh:
            fh.write("cycle,gnd_mean,gnd_max,accum_plastic_mean\n")
            for idx, (gm, gx, am) in enumerate(zip(gnd_means, gnd_max, accum_means), start=1):
                fh.write(f"{idx},{gm:.6e},{gx:.6e},{am:.6e}\n")
    print(payload)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
