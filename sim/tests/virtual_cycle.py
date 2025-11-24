"""
Driver script that runs a virtual cyclic tension test on a synthetic 111-oriented
single crystal and reports Paris/Coffin–Manson fits.

Updates:
- Supports multi-segment load path: 0 -> +max -> 0 -> -max -> 0 (per cycle).
- Adds failure cutoff based on crack_mean to stop early.
- Tracks plastic strain range per cycle (Δε_p surrogate) and crack increment.
- Fits Coffin–Manson using plastic range, and a Paris-like slope using crack growth per cycle.
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
    cycles: int = 3,               # 建议调试时先跑 1-2 个周期
    max_strain: float = 0.08,      # 拉伸峰值
    min_strain: float | None = None,  # 若为 None，使用对称 -max_strain
    segment_steps: int = 50,       # 每个子段（0->峰值）步数
    failure_threshold: float = 0.98,  # 平均裂纹达到此值提前终止
    csv_output: Path | None = None,
    data_output: Path | None = None,
    dump_dir: Path | None = None,
    vtk_dir: Path | None = None,
) -> Tuple[List[CycleResult], float, float]:
    
    # 1. 初始化
    # 【关键】无量纲设置：Spacing=1.0，材料/断裂参数在 energy.py 中已归一化
    grid = GridSpec(shape=(128, 64, 16), spacing=(1.0, 1.0, 1.0), periodic=(True, True, False))
    
    copper = CopperParameters()
    fracture = FractureParameters()
    pfc_params = PFCParameters()
    coupling = PFCCoupling(pfc_params, fracture, mode="density")
    energy = FreeEnergy(copper, fracture, coupling)
    
    builder = Cu111StructureBuilder(grid, defect_fraction=0.08, defect_amplitude=0.12)
    structure = builder.build(seed=42)
    
    mechanical = MechanicalEquilibriumSolver(grid, copper, structure.orientation)
    # 为避免 ψ 振幅溢出，开启温和截断
    # 应力耦合项：使用 von Mises 应力场放大 μ
    def mu_extra_from_stress(stress_vm):
        # 归一化后乘一个系数增强驱动
        max_vm = np.max(np.abs(stress_vm)) + 1e-12
        norm_vm = stress_vm / max_vm
        return 0.5 * norm_vm

    # pfc_extra_mu 将在 solver 内传入
    pfc = PFCEvolver(grid, pfc_params, dt=5e-3, clip=1.2)
    solver_cfg = SolverConfig(dt=5e-3, crack_relax=0.01, plastic_relax=0.2, mech_plastic_weight=0.9, dir_coupling=0.8)
    
    solver = AlternatingSolver(coupling, energy, mechanical, pfc, solver_cfg)
    solver.initialize_state(structure.orientation, seed=42)
    for key, value in structure.fields.items():
        solver.state[key] = value.copy()
    solver.state["history"] = np.zeros_like(structure.fields["psi"])

    results: List[CycleResult] = []
    current_strain = 0.0
    frame_id = 0

    # 2. 循环加载
    min_strain = -max_strain if min_strain is None else min_strain
    load_segments = [max_strain, 0.0, min_strain, 0.0]  # triangle: 0->+max->0->-max->0

    for cycle in range(1, cycles + 1):
        print(f"=== Starting Cycle {cycle} ===")
        energy_val = 0.0
        plastic_min, plastic_max = np.inf, -np.inf
        crack_prev = results[-1].crack_mean if results else 0.0
        
        for target in load_segments:
            target_start = current_strain
            target_end = target
            for step in range(1, segment_steps + 1):
                alpha = step / segment_steps
                current_strain = target_start + (target_end - target_start) * alpha
                energy_val = solver.step((current_strain, 0.0, 0.0))
                plast_mean = solver.state["plastic"].mean()
                plastic_min = min(plastic_min, plast_mean)
                plastic_max = max(plastic_max, plast_mean)

                if step % 5 == 0:
                    frame_id += 1
                    print(f"  Cycle {cycle} Substep {step}/{segment_steps} | Strain {current_strain:.4f}")
                    if vtk_dir:
                        vtk_dir.mkdir(parents=True, exist_ok=True)
                        write_vtk(
                            vtk_dir / f"anim_frame_{frame_id:05d}.vtk",
                            grid,
                            {
                                "crack": solver.state["crack"],
                                "plastic": solver.state["plastic"],
                                "plastic_vec": solver.state["plastic_vec"],
                                "psi": solver.state["psi"],
                                "displacement": solver.state["displacement"],
                                "stress_vm": solver.state["stress_vm"],
                            },
                            macro_strain=(current_strain, 0.0, 0.0),
                            deform_coordinates=True,
                        )

        crack_mean = solver.state["crack"].mean()
        plastic_mean = solver.state["plastic"].mean()
        plastic_range = max(plastic_max - plastic_min, 0.0)
        crack_delta = max(crack_mean - crack_prev, 0.0)
        results.append(CycleResult(cycle, energy_val, crack_mean, plastic_mean))

        if dump_dir:
            dump_dir.mkdir(parents=True, exist_ok=True)
            write_lammpstrj(
                dump_dir / f"virtual_cycle_{cycle:04d}.lammpstrj",
                grid,
                solver.state,
                cycle,
                macro_strain=(current_strain, 0.0, 0.0),
            )

        # Early stop if failed
        if crack_mean >= failure_threshold:
            print(f"[STOP] Crack mean {crack_mean:.3f} reached threshold {failure_threshold}.")
            break

    # 3. 后处理统计：Paris-like 与 Coffin–Manson
    plastic_ranges = []
    crack_deltas = []
    for i, r in enumerate(results):
        prev_crack = results[i - 1].crack_mean if i > 0 else r.crack_mean
        crack_deltas.append(max(r.crack_mean - prev_crack, 1e-9))
        # Plastic range per cycle was tracked above; reconstruct crude surrogate
        if i == 0:
            plastic_ranges.append(max(2 * r.plastic_mean, 1e-9))
        else:
            plastic_ranges.append(max(abs(r.plastic_mean - results[i - 1].plastic_mean), 1e-9))

    cycles_arr = np.arange(1, len(results) + 1, dtype=float)
    paris_coeff = 0.0
    if len(crack_deltas) > 1:
        mask = np.isfinite(crack_deltas) & (np.array(crack_deltas) > 0)
        if mask.any():
            paris_coeff = float(np.polyfit(np.log(np.array(crack_deltas)[mask]), np.log(cycles_arr[mask]), 1)[0])

    coffman = 0.0
    if len(plastic_ranges) > 0:
        pr = np.array(plastic_ranges)
        mask = pr > 0
        if mask.any():
            coffman = float(np.polyfit(np.log(pr[mask]), np.log(2 * cycles_arr[mask]), 1)[0])

    if csv_output:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", encoding="utf-8") as fh:
            fh.write("cycle,energy,crack_mean,plastic_mean,crack_delta,plastic_range\n")
            prev_crack = results[0].crack_mean if results else 0.0
            for r, pd, pr in zip(results, crack_deltas, plastic_ranges):
                fh.write(f"{r.cycle},{r.load:.6e},{r.crack_mean:.6e},{r.plastic_mean:.6e},{pd:.6e},{pr:.6e}\n")

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
        segment_steps=100
    )
