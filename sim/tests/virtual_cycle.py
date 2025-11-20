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
    cycles: int = 3,          # 建议调试时先跑 1-2 个周期
    max_strain: float = 0.08, # 8% 拉伸，足以产生位错
    strain_steps: int = 100,  # 每半周期分 100 步
    csv_output: Path | None = None,
    data_output: Path | None = None,
    dump_dir: Path | None = None,
    vtk_dir: Path | None = None,
) -> Tuple[List[CycleResult], float, float]:
    
    # 1. 初始化
    # 【关键】Spacing 改为 1.0 (无量纲) 以匹配 PFC
    grid = GridSpec(shape=(128, 64, 16), spacing=(1.0, 1.0, 1.0), periodic=(True, True, False))
    
    copper = CopperParameters()
    fracture = FractureParameters()
    pfc_params = PFCParameters()
    coupling = PFCCoupling(pfc_params, fracture, mode="density")
    energy = FreeEnergy(copper, fracture, coupling)
    
    builder = Cu111StructureBuilder(grid, defect_fraction=0.08, defect_amplitude=0.3)
    structure = builder.build(seed=42)
    
    mechanical = MechanicalEquilibriumSolver(grid, copper, structure.orientation)
    pfc = PFCEvolver(grid, pfc_params, dt=5e-3)
    solver_cfg = SolverConfig(dt=5e-3, crack_relax=0.01)
    
    solver = AlternatingSolver(coupling, energy, mechanical, pfc, solver_cfg)
    solver.initialize_state(structure.orientation, seed=42)
    for key, value in structure.fields.items():
        solver.state[key] = value.copy()
    solver.state["history"] = np.zeros_like(structure.fields["psi"])

    results: List[CycleResult] = []
    current_strain = 0.0
    frame_id = 0

    # 2. 循环加载
    for cycle in range(1, cycles + 1):
        print(f"=== Starting Cycle {cycle} ===")
        # 0 -> +Max -> -Max
        cycle_targets = [+max_strain, -max_strain]
        energy_val = 0.0
        
        for target_peak in cycle_targets:
            target_start = current_strain
            target_end = target_peak
            
            # 子步循环
            for step in range(1, strain_steps + 1):
                alpha = step / strain_steps
                current_strain = target_start + (target_end - target_start) * alpha
                
                # 求解
                energy_val = solver.step((current_strain, 0.0, 0.0))
                
                # 每 5 步输出一帧
                if step % 5 == 0:
                    frame_id += 1
                    print(f"  Cycle {cycle} Substep {step}/{strain_steps} | Strain {current_strain:.4f}")
                    if vtk_dir:
                        vtk_dir.mkdir(parents=True, exist_ok=True)
                        write_vtk(
                            vtk_dir / f"anim_frame_{frame_id:05d}.vtk",
                            grid,
                            {
                                "crack": solver.state["crack"],
                                "plastic": solver.state["plastic"],
                                "psi": solver.state["psi"],
                                "displacement": solver.state["displacement"],
                            },
                            macro_strain=(current_strain, 0.0, 0.0),
                            deform_coordinates=True,
                        )

        results.append(CycleResult(cycle, energy_val, solver.state["crack"].mean(), solver.state["plastic"].mean()))
        
        # 导出每个 Cycle 的汇总数据
        if dump_dir:
            dump_dir.mkdir(parents=True, exist_ok=True)
            write_lammpstrj(
                dump_dir / f"virtual_cycle_{cycle:04d}.lammpstrj",
                grid,
                solver.state,
                cycle,
                macro_strain=(current_strain, 0.0, 0.0),
            )

    # 3. 后处理统计 (修复了这里！)
    plastic_series = np.array([r.plastic_mean for r in results])
    # 处理 plastic_series 可能为空的情况
    if len(plastic_series) > 1:
        plastic_amplitudes = np.abs(np.diff(plastic_series, prepend=plastic_series[0]))
    else:
        plastic_amplitudes = np.zeros_like(plastic_series)

    crack_growth = np.array([r.crack_mean for r in results])
    
    paris_coeff = 0.0
    if len(crack_growth) > 1:
        dcrack = np.clip(np.diff(crack_growth), 1e-9, None)
        dN = np.ones_like(dcrack)
        mask = np.isfinite(dcrack) & (dcrack > 0)
        if mask.any():
            paris_coeff = float(np.polyfit(np.log(dcrack[mask]), np.log(dN[mask]), 1)[0])

    cycles_arr = np.arange(1, len(results) + 1, dtype=float)
    coffman = 0.0
    if len(plastic_amplitudes) > 0:
        coff_mask = plastic_amplitudes > 0
        if coff_mask.any():
            coffman = float(np.polyfit(np.log(plastic_amplitudes[coff_mask]), np.log(cycles_arr[coff_mask]), 1)[0])

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
    run_virtual_cycles(
        csv_output=Path("sim/tests/virtual_cycle.csv"),
        data_output=Path("sim/tests/virtual_cycle.data"),
        dump_dir=Path("sim/tests/virtual_cycle_lammpstrj"),
        vtk_dir=Path("sim/tests/virtual_cycle_vtk"),
        cycles=1,
        max_strain=0.08,
        strain_steps=100
    )
