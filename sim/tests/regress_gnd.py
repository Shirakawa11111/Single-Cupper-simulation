"""
GND (Nye tensor) regression check using a synthetic slip gradient.

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

from sim.dislocation import gnd_from_slip
from sim.energy import FractureParameters, PFCParameters, PFCCoupling
from sim.operators import GridSpec


@dataclass
class Thresholds:
    uniform_gnd_max: float = 1e-10
    gradient_gnd_min: float = 1e-5
    ratio_tol: float = 0.05


@dataclass
class Results:
    gnd_uniform_mean: float
    gnd_grad_mean: float
    gnd_double_mean: float
    ratio_double: float


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


def _test_gnd(slope: float = 1e-3) -> Results:
    grid = GridSpec(shape=(16, 8, 4), spacing=(1.0, 1.0, 1.0), periodic=(False, True, True))
    orientation = _make_orientation(grid)
    coupling = PFCCoupling(PFCParameters(), FractureParameters())
    n_slip = len(coupling.slip_systems)

    x = np.linspace(0.0, (grid.shape[0] - 1) * grid.spacing[0], grid.shape[0], dtype=float)
    ramp = x[:, None, None]

    gamma_uniform = np.zeros((n_slip,) + grid.shape, dtype=float)
    gamma_uniform[0] = 1.0
    rho0, _ = gnd_from_slip(gamma_uniform, coupling.slip_systems, orientation, grid, burgers=1.0)

    gamma_grad = np.zeros_like(gamma_uniform)
    gamma_grad[0] = slope * ramp
    rho1, _ = gnd_from_slip(gamma_grad, coupling.slip_systems, orientation, grid, burgers=1.0)

    gamma_double = np.zeros_like(gamma_uniform)
    gamma_double[0] = 2.0 * slope * ramp
    rho2, _ = gnd_from_slip(gamma_double, coupling.slip_systems, orientation, grid, burgers=1.0)

    mean0 = float(np.mean(rho0))
    mean1 = float(np.mean(rho1))
    mean2 = float(np.mean(rho2))
    ratio = mean2 / (mean1 + 1e-12)
    return Results(
        gnd_uniform_mean=mean0,
        gnd_grad_mean=mean1,
        gnd_double_mean=mean2,
        ratio_double=ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GND regression using synthetic slip gradient.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--slope", type=float, default=1e-3, help="Slip gradient magnitude.")
    parser.add_argument("--uniform-max", type=float, default=1e-10, help="Max allowed uniform GND mean.")
    parser.add_argument("--gradient-min", type=float, default=1e-5, help="Min allowed gradient GND mean.")
    parser.add_argument("--ratio-tol", type=float, default=0.05, help="Tolerance for doubling ratio.")
    args = parser.parse_args()

    thresholds = Thresholds(
        uniform_gnd_max=args.uniform_max,
        gradient_gnd_min=args.gradient_min,
        ratio_tol=args.ratio_tol,
    )
    failures: Dict[str, str] = {}
    t0 = perf_counter()
    results = _test_gnd(slope=args.slope)
    t1 = perf_counter()

    if results.gnd_uniform_mean > thresholds.uniform_gnd_max:
        failures["uniform_gnd"] = f"{results.gnd_uniform_mean:.6e} exceeds max"
    if results.gnd_grad_mean < thresholds.gradient_gnd_min:
        failures["gradient_gnd"] = f"{results.gnd_grad_mean:.6e} below min"
    if abs(results.ratio_double - 2.0) / 2.0 > thresholds.ratio_tol:
        failures["ratio_double"] = f"{results.ratio_double:.6e} exceeds tolerance"

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
