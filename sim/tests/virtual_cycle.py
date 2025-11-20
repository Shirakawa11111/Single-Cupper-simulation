"""
Driver script that runs a virtual cyclic tension test on a synthetic 111-oriented
single crystal and reports Paris/Coffin–Manson fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..energy import CopperParameters, FreeEnergy, FractureParameters, PFCParameters, PFCCoupling
from ..io import write_atomic_data, write_lammpstrj
from ..mechanics import MechanicalEquilibriumSolver
from ..operators import GridSpec
from ..pfc import PFCEvolver
from ..solver import AlternatingSolver, SolverConfig
from ..structure import Cu111StructureBuilder


@dataclass
class CycleResult:
    cycle: int
    load: float
    crack_mean: float
    plastic_mean: float


def run_virtual_cycles(
    cycles: int = 6,
    max_strain: float = 0.01,
    csv_output: Path | None = None,
    data_output: Path | None = None,
    dump_output: Path | None = None,
) -> Tuple[List[CycleResult], float, float]:
    grid = GridSpec(shape=(32, 32, 16), spacing=(5e-7, 5e-7, 5e-7), periodic=(True, True, False))
    copper = CopperParameters()
    fracture = FractureParameters()
    pfc_params = PFCParameters()
    coupling = PFCCoupling(pfc_params, fracture, mode="density")
    energy = FreeEnergy(copper, fracture, coupling)
    builder = Cu111StructureBuilder(grid, defect_fraction=0.08, defect_amplitude=0.3)
    structure = builder.build(seed=42)
    mechanical = MechanicalEquilibriumSolver(grid, copper, structure.orientation)
    pfc = PFCEvolver(grid, pfc_params, dt=5e-3)
    solver = AlternatingSolver(coupling, energy, mechanical, pfc)
    solver.initialize_state(structure.orientation, seed=42)
    for key, value in structure.fields.items():
        solver.state[key] = value.copy()
    solver.state["history"] = np.zeros_like(structure.fields["psi"])

    results: List[CycleResult] = []
    for cycle in range(1, cycles + 1):
        for sign in (+1, -1):
            target = (sign * max_strain, 0.0, 0.0)
            energy_val = solver.step(target)
        crack_mean = solver.state["crack"].mean()
        plastic_mean = solver.state["plastic"].mean()
        results.append(CycleResult(cycle, energy_val, crack_mean, plastic_mean))

    plastic_series = np.array([r.plastic_mean for r in results])
    plastic_amplitudes = np.abs(np.diff(plastic_series, prepend=plastic_series[0]))
    crack_growth = np.array([r.crack_mean for r in results])
    dcrack = np.clip(np.diff(crack_growth), 1e-9, None)
    dN = np.ones_like(dcrack)
    mask = np.isfinite(dcrack) & (dcrack > 0)
    paris_coeff = float(np.polyfit(np.log(dcrack[mask]), np.log(dN[mask]), 1)[0]) if mask.any() else 0.0
    cycles_arr = np.arange(1, len(results) + 1, dtype=float)
    coff_mask = plastic_amplitudes > 0
    coffman = (
        float(np.polyfit(np.log(plastic_amplitudes[coff_mask]), np.log(cycles_arr[coff_mask]), 1)[0])
        if coff_mask.any()
        else 0.0
    )

    if csv_output:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", encoding="utf-8") as fh:
            fh.write("cycle,energy,crack_mean,plastic_mean\n")
            for r in results:
                fh.write(f"{r.cycle},{r.load:.6e},{r.crack_mean:.6e},{r.plastic_mean:.6e}\n")

    if data_output:
        data_output.parent.mkdir(parents=True, exist_ok=True)
        write_atomic_data(data_output, grid)

    if dump_output:
        dump_output.parent.mkdir(parents=True, exist_ok=True)
        write_lammpstrj(dump_output, grid, solver.state, cycles)

    return results, paris_coeff, coffman


if __name__ == "__main__":
    res, paris, coff = run_virtual_cycles(
        csv_output=Path("sim/tests/virtual_cycle.csv"),
        data_output=Path("sim/tests/virtual_cycle.data"),
        dump_output=Path("sim/tests/virtual_cycle.lammpstrj"),
    )
    print(f"Completed {len(res)} cycles. Paris exponent ~ {paris:.3f}, Coffin-Manson ~ {coff:.3f}")
