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
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..analysis import crack_growth_rate, crack_length
from ..energy import CopperParameters, FreeEnergy, FractureParameters, PFCParameters, PFCCoupling
from ..io import write_atomic_data, write_lammpstrj, write_vtk
from ..mechanics import MechanicalConfig, MechanicalEquilibriumSolver
from ..operators import GridSpec
from ..pfc import PFCEvolver
from ..solver import AlternatingSolver, SolverConfig
from ..structure import Cu111StructureBuilder


def nondim_stress_to_gpa(stress_nd: np.ndarray, c11_GPa: float = 168.4) -> np.ndarray:
    """Convert nondimensional stress (σ* = σ/168.4 GPa) back to GPa."""
    return stress_nd * c11_GPa


def _sanitize_task(name: str) -> str:
    safe = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe) or "virtual_cycle"


def _default_run_dir(task: str) -> Path:
    date_str = date.today().isoformat()
    time_str = datetime.now().strftime("%H%M%S")
    safe_task = _sanitize_task(task)
    return Path("sim/tests/runs") / date_str / f"{safe_task}_{time_str}"


@dataclass
class CycleResult:
    cycle: int
    load: float
    crack_mean: float
    accum_plastic_mean: float
    crack_length: float = 0.0
    plastic_range: float = 0.0
    rss_peak_nd: float = 0.0
    rss_peak_nd_signed: float = 0.0


def run_virtual_cycles(
    cycles: int = 3,               # 建议调试时先跑 1-2 个周期
    max_strain: float = 0.02,      # 拉伸峰值（低应力/屈服平台场景）
    min_strain: float | None = None,  # 若为 None，使用对称 -max_strain
    segment_steps: int = 50,       # 每个子段（0->峰值）步数
    monotonic: bool = False,       # 若为 True，仅做 0->+max_strain 单调拉伸，便于 σ–ε 标定
    failure_threshold: float = 0.98,  # 平均裂纹达到此值提前终止
    csv_output: Path | None = None,
    analysis_csv: Path | None = None,  # 标准疲劳指标 CSV（a(N), da/dN, Δεp/2, rss_peak）
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
    plastic_relax: float = 0.075,       # 塑性松弛；与 PFCCoupling.flow_scale 共同控制屈服后软化速度
    poisson_ratio: float = 0.34,        # 泊松比，用于宏观应变的侧向收缩
    toughness_scale: float = 0.1,       # 韧性缩放因子 (<1 降低 Gc 促开裂；>1 提高韧性)
    stress_strain_csv: Path | None = None,  # 可选：逐步输出宏观应力-应变曲线
    crack_length_threshold: float = 0.95,   # 裂纹长度阈值
    crack_length_x0: float | None = None,   # 裂纹尖端参考点（None=自动）
    crack_length_axis: int = 0,             # 裂纹长度统计轴
    cycle_points: int | None = None,        # 每个完整循环的离散点数（覆盖 segment_steps）
    orientation_vector: tuple[float, float, float] = (1.0, 1.0, 1.0),  # 单晶取向，默认 [111]
    stable_window: int | None = None,       # 稳定判据：最近 N 个周期
    stable_tol: float = 0.02,               # 稳定判据：相对变化阈值
    stable_metrics: tuple[str, ...] = ("plastic_range", "rss_peak_nd"),
    stable_min_cycles: int | None = None,   # 最少周期后才开始判稳
    yield_tau: float | None = None,         # 覆盖屈服阈值（nd）
    flow_scale: float | None = None,        # 覆盖流动系数（nd）
    linear_hardening: float | None = None,  # 覆盖线性硬化（nd）
    visco_exponent: float | None = None,    # 覆盖黏性指数
    gamma0: float | None = None,            # 覆盖滑移剪切速率系数
    slip_exponent: float | None = None,     # 覆盖滑移指数 n
    h_iso: float | None = None,             # 覆盖各向同性硬化
    h_gnd: float | None = None,             # 覆盖 GND 硬化项
    mech_max_iters: int | None = None,      # 机械求解最大迭代
    mech_tol: float | None = None,          # 机械求解收敛阈值
    mech_outer_max_iters: int | None = None, # 单向分裂外循环
    mech_outer_tol: float | None = None,    # 单向分裂外循环阈值
    run_dir: Path | None = None,            # 输出根目录（默认按日期/任务自动生成）
    task: str = "virtual_cycle",            # 任务标签（用于命名输出目录）
    auto_output: bool = False,              # 若 True 且 run_dir 提供/默认，则自动填充输出文件
    pfc_active: bool = True,                # 是否演化 PFC
    gnd_active: bool = False,               # 是否输出 GND 诊断
    gnd_burgers: float = 1.0,               # Burgers 向量尺度（无量纲）
) -> Tuple[List[CycleResult], float, float]:
    
    # 1. 输出目录解析
    if run_dir is None and auto_output:
        run_dir = _default_run_dir(task)
    if run_dir is not None and auto_output:
        run_dir.mkdir(parents=True, exist_ok=True)
        if csv_output is None:
            csv_output = run_dir / "virtual_cycle.csv"
        if analysis_csv is None:
            analysis_csv = run_dir / "virtual_cycle_analysis.csv"
        if stress_strain_csv is None:
            stress_strain_csv = run_dir / "virtual_cycle_stress_strain.csv"
        if vtk_dir is None:
            vtk_dir = run_dir / "vtk"
        if dump_dir is None:
            dump_dir = run_dir / "lammpstrj"
        if data_output is None:
            data_output = run_dir / "virtual_cycle.data"
        if initial_vtk is None and vtk_dir is not None:
            initial_vtk = vtk_dir / "initial_seeded.vtk"

    # 2. 初始化
    # 【关键】无量纲设置：Spacing=1.0，stress* = stress_phys / 168.4 GPa (Cu c11)
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
    coupling_kwargs = {}
    if yield_tau is not None:
        coupling_kwargs["yield_tau"] = yield_tau
    if flow_scale is not None:
        coupling_kwargs["flow_scale"] = flow_scale
    if linear_hardening is not None:
        coupling_kwargs["linear_hardening"] = linear_hardening
    if visco_exponent is not None:
        coupling_kwargs["visco_exponent"] = visco_exponent
    if gamma0 is not None:
        coupling_kwargs["gamma0"] = gamma0
    elif flow_scale is not None:
        coupling_kwargs["gamma0"] = flow_scale
    if slip_exponent is not None:
        coupling_kwargs["slip_exponent"] = slip_exponent
    if h_iso is not None:
        coupling_kwargs["h_iso"] = h_iso
    if h_gnd is not None:
        coupling_kwargs["h_gnd"] = h_gnd
    coupling = PFCCoupling(pfc_params, fracture, mode="density", **coupling_kwargs)
    energy = FreeEnergy(copper, fracture, coupling)
    
    mech_cfg = MechanicalConfig()
    if mech_max_iters is not None:
        mech_cfg.max_iters = mech_max_iters
    if mech_tol is not None:
        mech_cfg.tol = mech_tol
    if mech_outer_max_iters is not None:
        mech_cfg.outer_max_iters = mech_outer_max_iters
    if mech_outer_tol is not None:
        mech_cfg.outer_tol = mech_outer_tol

    builder = Cu111StructureBuilder(
        grid,
        defect_fraction=0.08 if not defect_config else 0.0,
        defect_amplitude=0.12 if not defect_config else 0.0,
        defect_config=defect_config,
        orientation_vector=orientation_vector,
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
    
    mechanical = MechanicalEquilibriumSolver(
        grid, copper, structure.orientation, fracture_k=fracture.k, config=mech_cfg
    )
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
        coupling = PFCCoupling(pfc_params, fracture, mode="density", **coupling_kwargs)
        energy = FreeEnergy(copper, fracture, coupling)
        mechanical = MechanicalEquilibriumSolver(
            grid, copper, structure.orientation, fracture_k=fracture.k, config=mech_cfg
        )
        pfc = PFCEvolver(grid, pfc_params, dt=5e-3, clip=1.2)
        solver_cfg = SolverConfig(
            dt=5e-3,
            crack_relax=crack_relax,
            plastic_relax=plastic_relax,
            mech_plastic_weight=0.9,
            dir_coupling=dir_coupling,
            pfc_active=pfc_active,
            gnd_active=gnd_active,
            gnd_burgers=gnd_burgers,
        )
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
    # 统一裂纹长度参考点 x0
    crack_length_x0_resolved = crack_length_x0
    if crack_length_x0_resolved is None:
        if notch_box is not None:
            crack_length_x0_resolved = notch_box[0][1]
        else:
            crack_length_x0_resolved = crack_length(
                structure.fields["crack"],
                grid,
                axis=crack_length_axis,
                threshold=crack_length_threshold,
                x0=0.0,
            )
    # pfc_extra_mu 将在 solver 内传入
    pfc = PFCEvolver(grid, pfc_params, dt=5e-3, clip=1.2)
    solver_cfg = SolverConfig(
        dt=5e-3,
        crack_relax=crack_relax,
        plastic_relax=plastic_relax,
        mech_plastic_weight=0.9,
        dir_coupling=dir_coupling,
        pfc_active=pfc_active,
        gnd_active=gnd_active,
        gnd_burgers=gnd_burgers,
    )
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
    load_segments = [max_strain] if monotonic else [max_strain, 0.0, min_strain, 0.0]  # monotonic or triangle
    if cycle_points is not None:
        points_per_segment = max(1, int(round(cycle_points / len(load_segments))))
        if points_per_segment * len(load_segments) != cycle_points:
            print(
                f"[warn] cycle_points={cycle_points} not divisible by {len(load_segments)}; "
                f"using {points_per_segment} per segment -> {points_per_segment * len(load_segments)} points"
            )
        segment_steps = points_per_segment

    for cycle in range(1, cycles + 1):
        print(f"=== Starting Cycle {cycle} ===")
        energy_val = 0.0
        plastic_min, plastic_max = np.inf, -np.inf
        cycle_rss_peak = -np.inf
        
        for target in load_segments:
            target_start = current_strain
            target_end = target
            for step in range(1, segment_steps + 1):
                alpha = step / segment_steps
                current_strain = target_start + (target_end - target_start) * alpha
                macro = (current_strain, -poisson_ratio * current_strain, -poisson_ratio * current_strain)
                energy_val = solver.step(macro)
                plast_inst = solver.state.get("plastic_inst", solver.state["plastic"])
                plast_inst_mean = plast_inst.mean()
                accum_mean = solver.state.get("accum_plastic", solver.state["plastic"]).mean()
                plastic_min = min(plastic_min, plast_inst_mean)
                plastic_max = max(plastic_max, plast_inst_mean)
                stress_tensor = solver.state["stress"]
                stress_mean = np.mean(stress_tensor, axis=(0, 1, 2))
                stress_vm_mean = float(np.mean(solver.state.get("stress_vm", 0.0)))
                # RSS peak tracking (nd) per cycle
                rss_max, _, _, _, rss_signed_max = coupling.compute_rss(
                    stress_tensor,
                    backstress=solver.state.get("backstress"),
                    return_signed_max=True,
                    orientation=structure.orientation,
                )
                cycle_rss_peak = max(cycle_rss_peak, float(np.mean(rss_max)))
                cycle_rss_peak_signed = float(np.mean(rss_signed_max)) if rss_signed_max is not None else 0.0
                stress_strain_log.append(
                    (
                        current_strain,
                        stress_mean[0, 0],
                        stress_mean[1, 1],
                        stress_mean[2, 2],
                        stress_vm_mean,
                        plast_inst_mean,
                        accum_mean,
                        float(np.mean(rss_max)),
                        cycle_rss_peak_signed,
                    )
                )

                if step % 5 == 0:
                    frame_id += 1
                    print(f"  Cycle {cycle} Substep {step}/{segment_steps} | Strain {current_strain:.4f}")
                    if vtk_dir:
                        vtk_dir.mkdir(parents=True, exist_ok=True)
                        vtk_fields = {
                            "crack": solver.state["crack"],
                            "plastic": solver.state["plastic"],
                            "plastic_inst": solver.state.get("plastic_inst"),
                            "plastic_vec": solver.state["plastic_vec"],
                            "psi": solver.state["psi"],
                            "displacement": solver.state["displacement"],
                            "stress_vm": solver.state["stress_vm"],
                        }
                        gnd = solver.state.get("gnd_density")
                        if gnd is not None:
                            vtk_fields["gnd_density"] = gnd
                        tau_c = solver.state.get("tau_c")
                        if tau_c is not None:
                            vtk_fields["tau_c"] = tau_c
                        write_vtk(
                            vtk_dir / f"anim_frame_{frame_id:05d}.vtk",
                            grid,
                            vtk_fields,
                            macro_strain=macro,
                            deform_coordinates=True,
                        )

        crack_mean = solver.state["crack"].mean()
        accum_plastic_mean = solver.state.get("accum_plastic", solver.state["plastic"]).mean()
        crack_len = crack_length(
            solver.state["crack"],
            grid,
            axis=crack_length_axis,
            threshold=crack_length_threshold,
            x0=crack_length_x0_resolved,
        )
        plastic_range = max(plastic_max - plastic_min, 0.0)
        results.append(
            CycleResult(
                cycle=cycle,
                load=energy_val,
                crack_mean=crack_mean,
                accum_plastic_mean=accum_plastic_mean,
                crack_length=crack_len,
                plastic_range=plastic_range,
                rss_peak_nd=cycle_rss_peak,
                rss_peak_nd_signed=cycle_rss_peak_signed,
            )
        )

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

        # Early stop if stabilized
        if stable_window is not None and len(results) >= stable_window:
            if stable_min_cycles is None:
                stable_min_cycles = stable_window
            if cycle >= stable_min_cycles:
                rels = []
                for name in stable_metrics:
                    vals = np.array([getattr(r, name) for r in results[-stable_window:]], dtype=float)
                    mean = float(np.mean(vals))
                    rel = (float(np.max(vals)) - float(np.min(vals))) / (abs(mean) + 1e-12)
                    rels.append(rel)
                if all(r <= stable_tol for r in rels):
                    print(
                        f"[STOP] Stable over last {stable_window} cycles "
                        f"(metrics={stable_metrics}, tol={stable_tol})."
                    )
                    break

    # 3. 后处理统计：Paris-like 与 Coffin–Manson
    plastic_ranges = []
    crack_deltas = []
    a_vals = [r.crack_length for r in results]
    for i, r in enumerate(results):
        prev_a = a_vals[i - 1] if i > 0 else a_vals[i]
        crack_deltas.append(max(a_vals[i] - prev_a, 1e-9))
        plastic_ranges.append(max(r.plastic_range, 1e-9))

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
                fh.write("cycle,energy,crack_mean,accum_plastic_mean,crack_length,crack_delta,plastic_range\n")
                for r, pd, pr in zip(results, crack_deltas, plastic_ranges):
                    fh.write(
                        f"{r.cycle},{r.load:.6e},{r.crack_mean:.6e},"
                        f"{r.accum_plastic_mean:.6e},{r.crack_length:.6e},{pd:.6e},{pr:.6e}\n"
                    )
        if analysis_csv:
            analysis_csv.parent.mkdir(parents=True, exist_ok=True)
            cycles_out = np.array([r.cycle for r in results], dtype=float)
            a_arr = np.array(a_vals, dtype=float)
            if len(a_arr) > 1:
                da = crack_growth_rate(a_arr, cycles_out)
                da = np.concatenate(([0.0], da))
            else:
                da = np.zeros_like(a_arr)
            eps_p_half = 0.5 * np.array([r.plastic_range for r in results], dtype=float)
            rss_peak = np.array([r.rss_peak_nd for r in results], dtype=float)
            with analysis_csv.open("w", encoding="utf-8") as fh:
                fh.write("cycle,a,da_dN,eps_p_half,rss_peak_nd\n")
                for c, a, dadn, eph, rp in zip(cycles_out, a_arr, da, eps_p_half, rss_peak):
                    fh.write(f"{int(c)},{a:.6e},{dadn:.6e},{eph:.6e},{rp:.6e}\n")
        if stress_strain_csv and stress_strain_log:
            stress_strain_csv.parent.mkdir(parents=True, exist_ok=True)
            with stress_strain_csv.open("w", encoding="utf-8") as fh:
                fh.write(
                    "macro_strain,plastic_inst_mean,accum_plastic_mean,"
                    "sig_xx_nd,sig_yy_nd,sig_zz_nd,sig_vm_nd,"
                    "sig_xx_GPa,sig_yy_GPa,sig_zz_GPa,sig_vm_GPa,"
                    "rss_mean_nd,rss_mean_signed_nd\n"
                )
                for row in stress_strain_log:
                    plast_inst_mean = row[5]
                    accum_mean = row[6]
                    sig_xx_nd, sig_yy_nd, sig_zz_nd, sig_vm_nd = row[1], row[2], row[3], row[4]
                    rss_mean_nd = row[7]
                    rss_mean_signed_nd = row[8]
                    sig_xx_gpa, sig_yy_gpa, sig_zz_gpa, sig_vm_gpa = nondim_stress_to_gpa(
                        np.array([sig_xx_nd, sig_yy_nd, sig_zz_nd, sig_vm_nd])
                    )
                    fh.write(
                        f"{row[0]:.6e},{plast_inst_mean:.6e},{accum_mean:.6e},"
                        f"{sig_xx_nd:.6e},{sig_yy_nd:.6e},{sig_zz_nd:.6e},{sig_vm_nd:.6e},"
                        f"{sig_xx_gpa:.6e},{sig_yy_gpa:.6e},{sig_zz_gpa:.6e},{sig_vm_gpa:.6e},"
                        f"{rss_mean_nd:.6e},{rss_mean_signed_nd:.6e}\n"
                    )

    if data_output:
        data_output.parent.mkdir(parents=True, exist_ok=True)
        write_atomic_data(data_output, grid)

    return results, paris_coeff, coffman


def run_monotonic_tension(
    max_strain: float = 0.01,
    segment_steps: int = 100,
    **kwargs,
):
    """
    Convenience wrapper for a single-pass 0->+max_strain monotonic tension test.
    Useful for σ–ε calibration against experimental [111] Cu curves.
    """
    return run_virtual_cycles(
        cycles=1,
        max_strain=max_strain,
        min_strain=0.0,
        segment_steps=segment_steps,
        monotonic=True,
        **kwargs,
    )


def run_amplitude_sweep(
    amplitudes: list[float] | tuple[float, ...] = (2e-4, 5e-4, 1e-3),
    cycles: int = 80,
    steady_window: int = 10,
    **kwargs,
) -> list[tuple[float, float, float]]:
    """
    Sweep over strain amplitudes, run to (approximate) steady cyclic response,
    and return tuples (eps_amp, plastic_range_nd, rss_peak_MPa) averaged over the
    last `steady_window` cycles.
    """
    results_summary = []
    seg_steps = kwargs.pop("segment_steps", 50)
    for amp in amplitudes:
        res, _, _ = run_virtual_cycles(
            cycles=cycles,
            max_strain=amp,
            segment_steps=seg_steps,
            monotonic=False,
            **kwargs,
        )
        if not res:
            continue
        take = res[-steady_window:] if len(res) >= steady_window else res
        plast_ranges = [r.plastic_range for r in take]
        rss_peaks_nd = [r.rss_peak_nd for r in take]
        # convert rss_peak to MPa using 168.4 GPa scale
        rss_peaks_mpa = [r * 168.4e3 for r in rss_peaks_nd]
        plast_mean = float(np.mean(plast_ranges))
        rss_mean_mpa = float(np.mean(rss_peaks_mpa))
        results_summary.append((amp, plast_mean, rss_mean_mpa))
    return results_summary


if __name__ == "__main__":
    run_virtual_cycles(
        cycles=1,
        max_strain=0.02,
        segment_steps=100,
        defect_config=None,
        pre_relax_steps=0,
        pre_relax_strain=0.0,
        toughness_scale=0.1,
        task="virtual_cycle",
        auto_output=True,
    )
