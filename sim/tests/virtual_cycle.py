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
from ..io import write_atomic_data, write_lammpstrj, write_vtk
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
    cycles: int = 10,
    max_strain: float = 0.01,
    strain_steps: int = 400,
    csv_output: Path | None = None,
    data_output: Path | None = None,
    dump_dir: Path | None = None,
    vtk_dir: Path | None = None,
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
    solver_cfg = SolverConfig(dt=5e-3, crack_relax=0.1)
    solver = AlternatingSolver(coupling, energy, mechanical, pfc, solver_cfg)
    solver.initialize_state(structure.orientation, seed=42)
    for key, value in structure.fields.items():
        solver.state[key] = value.copy()
    solver.state["history"] = np.zeros_like(structure.fields["psi"])

    results: List[CycleResult] = []
    current_macro = 0.0
    for cycle in range(1, cycles + 1):
        for sign in (+1, -1):
            target_start = current_macro
            target_end = sign * max_strain
            # Apply the macroscopic strain in many small increments to mimic a realistic ramp.
            for step in range(1, strain_steps + 1):
                alpha = step / strain_steps
                target = target_start + (target_end - target_start) * alpha
                energy_val = solver.step((target, 0.0, 0.0))
            current_macro = target_end
        crack_mean = solver.state["crack"].mean()
        plastic_mean = solver.state["plastic"].mean()
        results.append(CycleResult(cycle, energy_val, crack_mean, plastic_mean))
        if dump_dir:
            dump_dir.mkdir(parents=True, exist_ok=True)
            write_lammpstrj(
                dump_dir / f"virtual_cycle_{cycle:04d}.lammpstrj",
                grid,
                solver.state,
                cycle,
            )
        if vtk_dir:
            vtk_dir.mkdir(parents=True, exist_ok=True)
            write_vtk(
        vtk_dir / f"virtual_cycle_{cycle:04d}.vtk",
                grid,
                {"crack": solver.state["crack"], "plastic": solver.state["plastic"], "psi": solver.state["psi"]},
            )

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

    return results, paris_coeff, coffman


if __name__ == "__main__":
    res, paris, coff = run_virtual_cycles(
        csv_output=Path("sim/tests/virtual_cycle.csv"),
        data_output=Path("sim/tests/virtual_cycle.data"),
        dump_dir=Path("sim/tests/virtual_cycle_lammpstrj"),
        vtk_dir=Path("sim/tests/virtual_cycle_vtk"),
    )
    print(f"Completed {len(res)} cycles. Paris exponent ~ {paris:.3f}, Coffin-Manson ~ {coff:.3f}")
