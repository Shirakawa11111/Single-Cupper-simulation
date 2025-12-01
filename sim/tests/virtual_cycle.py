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
    max_strain: float = 0.02,      # 拉伸峰值（低应力/屈服平台场景）
    min_strain: float | None = None,  # 若为 None，使用对称 -max_strain
    segment_steps: int = 50,       # 每个子段（0->峰值）步数
    failure_threshold: float = 0.98,  # 平均裂纹达到此值提前终止
    csv_output: Path | None = None,
    data_output: Path | None = None,
    dump_dir: Path | None = None,
    vtk_dir: Path | None = None,
    defect_config: dict | None = None,  # 缺陷播种配置（传给 Cu111StructureBuilder.defect_config）
    initial_vtk: Path | None = None,    # 可选：输出播种后的初始 VTK（含 deform_coordinates=False）
    pre_relax_steps: int = 0,           # 可选：载荷前的短暂预演化步数，形成更清晰的缺陷/滑移带
    pre_relax_strain: float = 0.0,      # 预演化时施加的宏观应变（通常为 0）
    notch_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,  # 可选：在区域内预置裂纹/notch
    notch_crack_value: float = 0.6,     # notch 区域内的初始裂纹值
    stress_mu_weight: float = 0.5,      # von Mises 应力归一化后乘此系数作为 μ_extra，<=0 则关闭
    crack_relax: float = 0.05,          # 裂纹松弛系数（默认较高便于萌生）
    dir_coupling: float = 1.0,          # 方向性增益
    plastic_relax: float = 0.2,         # 塑性松弛
    poisson_ratio: float = 0.34,        # 泊松比，用于宏观应变的侧向收缩
    toughness_scale: float = 0.1,       # 韧性缩放因子 (<1 降低 Gc 促开裂；>1 提高韧性)
    stress_strain_csv: Path | None = None,  # 可选：逐步输出宏观应力-应变曲线
) -> Tuple[List[CycleResult], float, float]:
    
    # 1. 初始化
    # 【关键】无量纲设置：Spacing=1.0，材料/断裂参数在 energy.py 中已归一化
    grid = GridSpec(shape=(128, 64, 16), spacing=(1.0, 1.0, 1.0), periodic=(True, True, False))
    
    copper = CopperParameters()
    base_fracture = FractureParameters()
    fracture = FractureParameters(
        gc=base_fracture.gc * toughness_scale,
        l0=base_fracture.l0,
        k=base_fracture.k,
        epsilon_half=base_fracture.epsilon_half,
        gres=base_fracture.gres * toughness_scale,
    )
    pfc_params = PFCParameters()
    coupling = PFCCoupling(pfc_params, fracture, mode="density")
    energy = FreeEnergy(copper, fracture, coupling)
    
    builder = Cu111StructureBuilder(
        grid,
        defect_fraction=0.08 if not defect_config else 0.0,
        defect_amplitude=0.12 if not defect_config else 0.0,
        defect_config=defect_config,
    )
    structure = builder.build(seed=42)

    # 可选：预置 notch/裂纹种子
    if notch_box is not None:
        (x0, x1), (y0, y1), (z0, z1) = notch_box
        dx, dy, dz = grid.spacing
        nx, ny, nz = grid.shape
        xs = np.linspace(0, dx * (nx - 1), nx)
        ys = np.linspace(0, dy * (ny - 1), ny)
        zs = np.linspace(0, dz * (nz - 1), nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        mask = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1)
        structure.fields["crack"][mask] = np.clip(notch_crack_value, 0.0, 1.0)
    
    mechanical = MechanicalEquilibriumSolver(grid, copper, structure.orientation)
    if initial_vtk:
        initial_vtk.parent.mkdir(parents=True, exist_ok=True)
        write_vtk(initial_vtk, grid, structure.fields, macro_strain=(0.0, 0.0, 0.0), deform_coordinates=False)

    # 短暂预演化：在载荷前让 ψ/裂纹/塑性更“尖锐”
    if pre_relax_steps > 0:
        print(f"[Pre-relax] steps={pre_relax_steps}, strain={pre_relax_strain}")
        # 初始化一次 solver 与状态
        copper = CopperParameters()
        base_fracture = FractureParameters()
        fracture = FractureParameters(
            gc=base_fracture.gc * toughness_scale,
            l0=base_fracture.l0,
            k=base_fracture.k,
            epsilon_half=base_fracture.epsilon_half,
            gres=base_fracture.gres * toughness_scale,
        )
        pfc_params = PFCParameters()
        coupling = PFCCoupling(pfc_params, fracture, mode="density")
        energy = FreeEnergy(copper, fracture, coupling)
        mechanical = MechanicalEquilibriumSolver(grid, copper, structure.orientation)
        pfc = PFCEvolver(grid, pfc_params, dt=5e-3, clip=1.2)
        solver_cfg = SolverConfig(dt=5e-3, crack_relax=crack_relax, plastic_relax=plastic_relax, mech_plastic_weight=0.9, dir_coupling=dir_coupling)
        mu_extra = None
        if stress_mu_weight > 0:
            mu_extra = lambda svm: stress_mu_weight * svm / (np.max(np.abs(svm)) + 1e-12)
        solver = AlternatingSolver(
            coupling, energy, mechanical, pfc, solver_cfg, mu_extra_from_stress=mu_extra, grain_mask=structure.grain_mask
        )
        solver.initialize_state(structure.orientation, seed=42)
        for key, value in structure.fields.items():
            solver.state[key] = value.copy()
        solver.state["history"] = np.zeros_like(structure.fields["psi"])
        for s in range(pre_relax_steps):
            solver.step((pre_relax_strain, 0.0, 0.0))
        # 将预演化后的场作为新的初值
        for key in structure.fields.keys():
            structure.fields[key] = solver.state[key].copy()
        if initial_vtk:
            write_vtk(initial_vtk.with_name(initial_vtk.stem + "_prerelax.vtk"), grid, structure.fields, macro_strain=(0.0, 0.0, 0.0), deform_coordinates=False)
    # pfc_extra_mu 将在 solver 内传入
    pfc = PFCEvolver(grid, pfc_params, dt=5e-3, clip=1.2)
    solver_cfg = SolverConfig(dt=5e-3, crack_relax=crack_relax, plastic_relax=plastic_relax, mech_plastic_weight=0.9, dir_coupling=dir_coupling)
    mu_extra = None
    if stress_mu_weight > 0:
        mu_extra = lambda svm: stress_mu_weight * svm / (np.max(np.abs(svm)) + 1e-12)
    solver = AlternatingSolver(
        coupling, energy, mechanical, pfc, solver_cfg, mu_extra_from_stress=mu_extra, grain_mask=structure.grain_mask
    )
    solver.initialize_state(structure.orientation, seed=42)
    for key, value in structure.fields.items():
        solver.state[key] = value.copy()
    solver.state["history"] = np.zeros_like(structure.fields["psi"])
    solver.state["accum_plastic"] = solver.state.get("plastic", np.zeros_like(structure.fields["psi"])).copy()

    results: List[CycleResult] = []
    current_strain = 0.0
    frame_id = 0
    stress_strain_log = []

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
                macro = (current_strain, -poisson_ratio * current_strain, -poisson_ratio * current_strain)
                energy_val = solver.step(macro)
                plast_mean = solver.state["plastic"].mean()
                plastic_min = min(plastic_min, plast_mean)
                plastic_max = max(plastic_max, plast_mean)
                stress_tensor = solver.state["stress"]
                stress_mean = np.mean(stress_tensor, axis=(0, 1, 2))
                stress_vm_mean = float(np.mean(solver.state.get("stress_vm", 0.0)))
                stress_strain_log.append(
                    (current_strain, stress_mean[0, 0], stress_mean[1, 1], stress_mean[2, 2], stress_vm_mean)
                )

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
                            macro_strain=macro,
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
                macro_strain=macro,
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
        if stress_strain_csv and stress_strain_log:
            stress_strain_csv.parent.mkdir(parents=True, exist_ok=True)
            with stress_strain_csv.open("w", encoding="utf-8") as fh:
                fh.write("macro_strain,sig_xx,sig_yy,sig_zz,sig_vm\n")
                for row in stress_strain_log:
                    fh.write(",".join(f"{v:.6e}" for v in row) + "\n")

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
        max_strain=0.02,
        segment_steps=100,
        defect_config=None,
        initial_vtk=Path("sim/tests/virtual_cycle_vtk/initial_seeded.vtk"),
        pre_relax_steps=0,
        pre_relax_strain=0.0,
        stress_strain_csv=Path("sim/tests/virtual_cycle_stress_strain.csv"),
        toughness_scale=0.1,
    )
