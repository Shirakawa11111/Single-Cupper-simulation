"""
Quick smoke test: initialize seeded defects, run a few alternating-solver steps, dump VTK/LAMMPS.

Usage:
    PYTHONPATH=. python -m sim.tests.smoke_seeded
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..energy import CopperParameters, FractureParameters, FreeEnergy, PFCParameters, PFCCoupling
from ..io import write_lammpstrj, write_vtk
from ..mechanics import MechanicalEquilibriumSolver
from ..operators import GridSpec
from ..pfc import PFCEvolver
from ..solver import AlternatingSolver, SolverConfig
from ..structure import Cu111StructureBuilder


def main() -> None:
    grid = GridSpec(shape=(64, 32, 16), spacing=(0.1, 0.1, 0.1), periodic=(True, True, False))

    defect_cfg = {
        "seed_density": 1e14,
        "region_bounds": ((0, 6.4), (0, 3.2), (0, 1.6)),
        "type_probabilities": {"vacancy": 0.34, "interstitial": 0.33, "dislocation": 0.33},
        "orient_mode": "slip_system",
        "max_seeds": 16384,
        "sigma_defect": 0.1,
        "sigma_normal": 0.2,
    }

    builder = Cu111StructureBuilder(
        grid,
        defect_config=defect_cfg,
        defect_fraction=0.0,
        defect_amplitude=0.0,
        noise=1e-3,
    )
    structure = builder.build(seed=42)

    copper = CopperParameters()
    fracture = FractureParameters()
    pfc_params = PFCParameters()
    coupling = PFCCoupling(pfc_params, fracture, mode="density")
    energy = FreeEnergy(copper, fracture, coupling)

    mechanical = MechanicalEquilibriumSolver(grid, copper, structure.orientation, fracture_k=fracture.k)
    pfc = PFCEvolver(grid, pfc_params, dt=5e-3, clip=1.2)
    solver_cfg = SolverConfig(dt=5e-3, crack_relax=0.01, plastic_relax=0.2, mech_plastic_weight=0.9, dir_coupling=0.8)
    solver = AlternatingSolver(coupling, energy, mechanical, pfc, solver_cfg)
    solver.initialize_state(structure.orientation, seed=42)
    for key, value in structure.fields.items():
        solver.state[key] = value.copy()
    solver.state["history"] = np.zeros_like(structure.fields["psi"])

    steps = 5
    max_strain = 0.02
    out_dir = Path("sim/tests/seeded_cu_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    lammpstrj_path = out_dir / "seeded_cu_smoke.lammpstrj"

    for s in range(1, steps + 1):
        eps = max_strain * s / steps
        energy_val = solver.step((eps, 0.0, 0.0))
        plastic_mean = solver.state.get("accum_plastic", solver.state["plastic"]).mean()
        print(
            f"step {s}/{steps} strain={eps:.4f} "
            f"energy={energy_val:.4e} crack_mean={solver.state['crack'].mean():.4e} "
            f"plastic_mean={plastic_mean:.4e}"
        )
        # per-step VTK (deformed coords)
        write_vtk(
            out_dir / f"seeded_cu_smoke_step{s:03d}.vtk",
            grid,
            solver.state,
            macro_strain=(eps, 0.0, 0.0),
            deform_coordinates=True,
        )

    # final snapshots
    write_vtk(out_dir / "seeded_cu_smoke_final_deformed.vtk", grid, solver.state, macro_strain=(max_strain, 0.0, 0.0), deform_coordinates=True)
    write_vtk(out_dir / "seeded_cu_smoke_final_undeformed.vtk", grid, solver.state, macro_strain=None, deform_coordinates=False)
    write_lammpstrj(lammpstrj_path, grid, solver.state, timestep=steps, macro_strain=(max_strain, 0.0, 0.0))
    print(f"Wrote VTK sequence to {out_dir} and LAMMPS dump to {lammpstrj_path}")


if __name__ == "__main__":
    main()
