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

from dataclasses import dataclass, replace
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


def nondim_stress_to_gpa(stress_nd: np.ndarray, stress_ref_gpa: float = 168.4) -> np.ndarray:
    """Convert nondimensional stress (σ* = σ/stress_ref) back to GPa."""
    return stress_nd * float(stress_ref_gpa)


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
    gnd_mean: float = 0.0
    gnd_max: float = 0.0
    crack_length: float = 0.0
    plastic_range: float = 0.0
    rss_peak_nd: float = 0.0
    rss_peak_nd_signed: float = 0.0
    crack_p95: float = 0.0
    crack_p99: float = 0.0
    crack_localization_index: float = 0.0
    energy_elastic_mean: float = 0.0
    energy_pfc_mean: float = 0.0
    energy_crack_mean: float = 0.0
    energy_total_density_mean: float = 0.0


def run_virtual_cycles(
    cycles: int = 3,               # 建议调试时先跑 1-2 个周期
    max_strain: float = 0.02,      # 拉伸峰值（低应力/屈服平台场景）
    max_strain_schedule: tuple[float, ...] | list[float] | None = None,  # 分级续算应变幅（按周期覆盖 max_strain）
    min_strain: float | None = None,  # 若为 None，使用对称 -max_strain
    segment_steps: int = 50,       # 每个子段（0->峰值）步数
    print_interval: int = 5,       # 进度打印间隔（步）
    vtk_interval: int = 5,         # VTK 输出间隔（步）
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
    arc_notch: dict | None = None,      # 可选：圆弧切口配置（圆心可在试样外）
    arc_notch_min_cells_x: int = 2,     # 弧形切口 x 向最小可解析网格数（过窄会自动扩展）
    notch_crack_value: float = 0.6,     # notch 区域内的初始裂纹值
    stress_mu_weight: float = 0.5,      # von Mises 应力归一化后乘此系数作为 μ_extra，<=0 则关闭
    crack_relax: float = 0.05,          # 裂纹松弛系数（默认较高便于萌生）
    localization_trigger_threshold: float = 0.0,  # 局部化触发：crack>=阈值进入高驱动区（<=0 关闭）
    localization_trigger_boost: float = 1.0,      # 局部化区 history 放大系数
    localization_background_scale: float = 1.0,   # 非局部化区 history 缩放系数
    crack_tol: float | None = None,     # 裂纹 CG 收敛阈值
    crack_max_iters: int | None = None, # 裂纹 CG 最大迭代
    crack_accept_rel_residual: float | None = None, # 裂纹 CG 相对残差验收阈值
    crack_accept_incomplete: bool | None = None, # 裂纹 CG 是否接受 incomplete 结果
    dir_coupling: float = 1.0,          # 方向性增益
    plastic_relax: float = 0.075,       # 塑性松弛；与 PFCCoupling.flow_scale 共同控制屈服后软化速度
    coupling_inner_iters: int = 3,      # 机械-塑性增量内固定点迭代次数
    coupling_min_iters: int = 1,        # 最小内迭代次数
    coupling_tol: float = 5e-4,         # 塑性张量内迭代收敛阈值
    coupling_stress_tol: float = 5e-4,  # 应力内迭代收敛阈值
    post_crack_mech_correction: bool = True,  # 裂纹更新后再做一次力学校正
    poisson_ratio: float = 0.34,        # 泊松比，用于宏观应变的侧向收缩
    toughness_scale: float = 0.1,       # 韧性缩放因子 (<1 降低 Gc 促开裂；>1 提高韧性)
    stress_strain_csv: Path | None = None,  # 可选：逐步输出宏观应力-应变曲线
    crack_length_threshold: float = 0.95,   # 裂纹长度阈值
    crack_length_x0: float | None = None,   # 裂纹尖端参考点（None=自动）
    crack_length_axis: int = 0,             # 裂纹长度统计轴
    cycle_points: int | None = None,        # 每个完整循环的离散点数（覆盖 segment_steps）
    max_strain_step: float | None = None,   # 单步应变增量上限（用于大应变幅稳定积分）
    adaptive_substep_retry: bool = True,    # 机械步拒绝时自动二分子步重试
    adaptive_substep_max_splits: int = 8,   # 单个名义步最大二分次数
    adaptive_substep_min_increment: float = 1e-8,  # 最小可尝试应变增量
    stress_ref_gpa: float = 168.4,          # 无量纲应力转物理量的参考应力
    orientation_vector: tuple[float, float, float] = (1.0, 1.0, 1.0),  # 单晶取向，默认 [111]
    random_seed: int = 42,                  # 随机种子（结构噪声/缺陷播种与初始化）
    grid_shape: tuple[int, int, int] | None = None,
    grid_spacing: tuple[float, float, float] | None = None,
    grid_periodic: tuple[bool, bool, bool] | None = None,
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
    yield_tau_inf: float | None = None,     # 覆盖饱和 CRSS（Voce tau_inf, nd）
    h0_iso: float | None = None,            # 覆盖 Voce 初始硬化模量 h0（nd）
    h_gnd: float | None = None,             # 覆盖 GND 硬化项
    kin_c: float | None = None,             # 覆盖运动硬化 C（Armstrong-Frederick）
    kin_d: float | None = None,             # 覆盖运动硬化 D（Armstrong-Frederick）
    kin_c2: float | None = None,            # 覆盖第二背应力 C2（双背应力）
    kin_d2: float | None = None,            # 覆盖第二背应力 D2（双背应力）
    c11: float | None = None,               # 覆盖铜 c11（无量纲）
    c12: float | None = None,               # 覆盖铜 c12（无量纲）
    c44: float | None = None,               # 覆盖铜 c44（无量纲）
    mech_max_iters: int | None = None,      # 机械求解最大迭代
    mech_tol: float | None = None,          # 机械求解收敛阈值
    mech_outer_max_iters: int | None = None, # 单向分裂外循环
    mech_outer_tol: float | None = None,    # 单向分裂外循环阈值
    mech_regularization: float | None = None, # 机械求解线性正则
    mech_accept_rel_residual: float | None = None, # 迭代残差验收阈值
    mech_accept_incomplete_cg: bool | None = None, # 是否接受 info>0 的有限 CG 解
    mech_accept_incomplete_without_residual: bool | None = None, # 是否接受 residual 非有限的 incomplete CG 解
    mech_enable_gmres_fallback: bool | None = None, # 是否启用 GMRES 回退
    mech_gmres_restart: int | None = None,  # GMRES restart
    mech_gmres_maxiter: int | None = None,  # GMRES maxiter
    mech_solution_abs_limit: float | None = None, # 迭代解绝对值上限
    mech_clip_solution_on_limit: bool | None = None, # 解幅值超限时是否裁剪而非拒收
    mech_unilateral: bool | None = None,    # 是否启用单向分裂（False=线性力学）
    mech_unilateral_mode: str | None = None,  # 单向分裂模式: spectral/volumetric
    mech_preconditioner: str | None = None,  # 机械线性求解预条件: none/jacobi
    mech_preconditioner_floor: float | None = None,  # Jacobi 对角下限
    mech_preconditioner_g_min: float | None = None,  # Jacobi 裂纹退化下限
    mech_displacement_bc_x: bool | None = None,  # X 向位移边界驱动（x=0/L）
    mech_displacement_bc_penalty: float | None = None,  # 位移边界惩罚系数
    mech_displacement_bc_anchor_lateral: bool | None = None,  # 锚定一个节点的 y/z 消除刚体模态
    mech_displacement_bc_hard: bool | None = None,  # 以强约束方式施加位移边界（非罚函数）
    mech_nondim_kinematics: bool | None = None,  # 力学坐标/位移无量纲一致化（改善条件数）
    run_dir: Path | None = None,            # 输出根目录（默认按日期/任务自动生成）
    task: str = "virtual_cycle",            # 任务标签（用于命名输出目录）
    auto_output: bool = False,              # 若 True 且 run_dir 提供/默认，则自动填充输出文件
    export_energy_fields: bool = False,     # 若 True，输出能量密度/裂纹驱动力场到 VTK 并记录周期统计
    pfc_active: bool = True,                # 是否演化 PFC
    pfc_fft_threads: int | None = None,      # PFC FFT 线程数（None=默认）
    pfc_use_pyfftw: bool | None = None,      # 是否使用 pyFFTW（None=默认）
    gnd_active: bool = False,               # 是否输出 GND 诊断
    gnd_burgers: float = 1.0,               # Burgers 向量尺度（无量纲）
    diagnostics_out: dict[str, float | int | bool] | None = None,  # 可选：回填数值稳定诊断
    state_out: dict[str, np.ndarray] | None = None,  # 可选：导出最终场状态（用于后处理）
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
    shape = grid_shape or (128, 64, 16)
    spacing = grid_spacing or (1.0, 1.0, 1.0)
    periodic = grid_periodic or (True, True, False)
    grid = GridSpec(shape=shape, spacing=spacing, periodic=periodic)
    
    copper_overrides: dict[str, float] = {}
    if c11 is not None:
        copper_overrides["c11"] = float(c11)
    if c12 is not None:
        copper_overrides["c12"] = float(c12)
    if c44 is not None:
        copper_overrides["c44"] = float(c44)
    if any(v <= 0.0 for v in copper_overrides.values()):
        raise ValueError("c11/c12/c44 overrides must be positive.")
    copper = replace(CopperParameters(), **copper_overrides)
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
    if yield_tau_inf is not None:
        coupling_kwargs["yield_tau_inf"] = yield_tau_inf
    if h0_iso is not None:
        coupling_kwargs["h0_iso"] = h0_iso
    if h_gnd is not None:
        coupling_kwargs["h_gnd"] = h_gnd
    if kin_c is not None:
        coupling_kwargs["kin_c"] = kin_c
    if kin_d is not None:
        coupling_kwargs["kin_d"] = kin_d
    if kin_c2 is not None:
        coupling_kwargs["kin_c2"] = kin_c2
    if kin_d2 is not None:
        coupling_kwargs["kin_d2"] = kin_d2
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
    if mech_regularization is not None:
        mech_cfg.regularization = mech_regularization
    if mech_accept_rel_residual is not None:
        mech_cfg.accept_rel_residual = mech_accept_rel_residual
    if mech_accept_incomplete_cg is not None:
        mech_cfg.accept_incomplete_cg = mech_accept_incomplete_cg
    if mech_accept_incomplete_without_residual is not None:
        mech_cfg.accept_incomplete_without_residual = mech_accept_incomplete_without_residual
    if mech_enable_gmres_fallback is not None:
        mech_cfg.enable_gmres_fallback = mech_enable_gmres_fallback
    if mech_gmres_restart is not None:
        mech_cfg.gmres_restart = mech_gmres_restart
    if mech_gmres_maxiter is not None:
        mech_cfg.gmres_maxiter = mech_gmres_maxiter
    if mech_solution_abs_limit is not None:
        mech_cfg.solution_abs_limit = mech_solution_abs_limit
    if mech_clip_solution_on_limit is not None:
        mech_cfg.clip_solution_on_limit = mech_clip_solution_on_limit
    if mech_unilateral is not None:
        mech_cfg.unilateral = bool(mech_unilateral)
    if mech_unilateral_mode is not None:
        if mech_unilateral_mode not in ("spectral", "volumetric"):
            raise ValueError("mech_unilateral_mode must be 'spectral' or 'volumetric'.")
        mech_cfg.unilateral_mode = mech_unilateral_mode
    if mech_preconditioner is not None:
        if mech_preconditioner not in ("none", "jacobi"):
            raise ValueError("mech_preconditioner must be 'none' or 'jacobi'.")
        mech_cfg.preconditioner = mech_preconditioner
    if mech_preconditioner_floor is not None:
        mech_cfg.preconditioner_floor = mech_preconditioner_floor
    if mech_preconditioner_g_min is not None:
        mech_cfg.preconditioner_g_min = mech_preconditioner_g_min
    if mech_displacement_bc_x is not None:
        mech_cfg.displacement_bc_x = bool(mech_displacement_bc_x)
    if mech_displacement_bc_penalty is not None:
        mech_cfg.displacement_bc_penalty = float(mech_displacement_bc_penalty)
    if mech_displacement_bc_anchor_lateral is not None:
        mech_cfg.displacement_bc_anchor_lateral = bool(mech_displacement_bc_anchor_lateral)
    if mech_displacement_bc_hard is not None:
        mech_cfg.displacement_bc_hard = bool(mech_displacement_bc_hard)
    if mech_nondim_kinematics is not None:
        mech_cfg.nondim_kinematics = bool(mech_nondim_kinematics)

    builder = Cu111StructureBuilder(
        grid,
        defect_fraction=0.08 if not defect_config else 0.0,
        defect_amplitude=0.12 if not defect_config else 0.0,
        defect_config=defect_config,
        orientation_vector=orientation_vector,
    )
    structure = builder.build(seed=int(random_seed))

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
    if arc_notch is not None:
        if not isinstance(arc_notch, dict):
            raise ValueError("arc_notch must be a mapping when provided.")
        dx, dy, dz = grid.spacing
        nx, ny, nz = grid.shape
        xs = np.linspace(0, dx * (nx - 1), nx)
        ys = np.linspace(0, dy * (ny - 1), ny)
        zs = np.linspace(0, dz * (nz - 1), nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        center = arc_notch.get("center", [0.5 * dx * (nx - 1), dy * ny])
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError("arc_notch.center must be [x_center, y_center].")
        cx, cy = float(center[0]), float(center[1])
        radius = float(arc_notch.get("radius", min(dx * (nx - 1), dy * (ny - 1)) * 0.5))
        if radius <= 0.0:
            raise ValueError("arc_notch.radius must be > 0.")
        surface = str(arc_notch.get("surface", "y_max"))
        crack_val = float(arc_notch.get("crack_value", notch_crack_value))
        z_range = arc_notch.get("z_range", None)
        x_range = arc_notch.get("x_range", None)

        if surface == "y_max":
            rad2 = radius * radius - (X - cx) ** 2
            valid = rad2 >= 0.0
            y_arc = cy - np.sqrt(np.clip(rad2, 0.0, None))
            y_top = dy * (ny - 1)
            mask = valid & (Y >= y_arc) & (Y <= y_top + 1e-12)
        elif surface == "y_min":
            rad2 = radius * radius - (X - cx) ** 2
            valid = rad2 >= 0.0
            y_arc = cy + np.sqrt(np.clip(rad2, 0.0, None))
            y_bot = 0.0
            mask = valid & (Y <= y_arc) & (Y >= y_bot - 1e-12)
        else:
            raise ValueError("arc_notch.surface must be 'y_max' or 'y_min'.")

        if isinstance(z_range, (list, tuple)) and len(z_range) == 2:
            z0, z1 = float(z_range[0]), float(z_range[1])
            mask &= (Z >= min(z0, z1)) & (Z <= max(z0, z1))
        if isinstance(x_range, (list, tuple)) and len(x_range) == 2:
            x0, x1 = float(x_range[0]), float(x_range[1])
            x0, x1 = min(x0, x1), max(x0, x1)
            min_cells_cfg = arc_notch.get("min_cells_x", arc_notch_min_cells_x)
            min_cells = max(1, int(min_cells_cfg))
            min_width = float(min_cells) * dx
            width = x1 - x0
            if width + 1e-15 < min_width:
                xc = 0.5 * (x0 + x1)
                half = 0.5 * min_width
                x_min_domain = 0.0
                x_max_domain = dx * (nx - 1)
                x0 = max(x_min_domain, xc - half)
                x1 = min(x_max_domain, xc + half)
                if (x1 - x0) + 1e-15 < min_width:
                    if x0 <= x_min_domain + 1e-12:
                        x1 = min(x_max_domain, x_min_domain + min_width)
                    else:
                        x0 = max(x_min_domain, x_max_domain - min_width)
                x0, x1 = min(x0, x1), max(x0, x1)
                print(
                    f"[notch] arc_notch.x_range expanded to resolvable width: "
                    f"cells>={min_cells}, dx={dx:.3e}, x_range=[{x0:.3e}, {x1:.3e}]"
                )
            mask &= (X >= min(x0, x1)) & (X <= max(x0, x1))

        masked_pts = int(np.count_nonzero(mask))
        if masked_pts == 0:
            print(
                "[notch] warning: arc_notch produced zero seeded cells. "
                "Increase radius/x_range or reduce grid spacing."
            )

        structure.fields["crack"][mask] = np.clip(crack_val, 0.0, 1.0)
    
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
        pfc = PFCEvolver(
            grid,
            pfc_params,
            dt=5e-3,
            clip=1.2,
            fft_threads=pfc_fft_threads,
            use_pyfftw=(pfc_use_pyfftw if pfc_use_pyfftw is not None else True),
        )
        solver_cfg = SolverConfig(
            dt=5e-3,
            crack_relax=crack_relax,
            history_localization_crack_threshold=localization_trigger_threshold,
            history_localization_boost=localization_trigger_boost,
            history_background_scale=localization_background_scale,
            crack_tol=crack_tol if crack_tol is not None else 1e-6,
            crack_max_iters=crack_max_iters if crack_max_iters is not None else 400,
            crack_accept_rel_residual=(
                crack_accept_rel_residual if crack_accept_rel_residual is not None else 5e-3
            ),
            crack_accept_incomplete=(
                crack_accept_incomplete if crack_accept_incomplete is not None else True
            ),
            plastic_relax=plastic_relax,
            coupling_inner_iters=max(1, int(coupling_inner_iters)),
            coupling_min_iters=max(1, int(coupling_min_iters)),
            coupling_tol=float(coupling_tol),
            coupling_stress_tol=float(coupling_stress_tol),
            post_crack_mech_correction=bool(post_crack_mech_correction),
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
        solver.initialize_state(structure.orientation, seed=int(random_seed))
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
        print("[Pre-relax] done")
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
    pfc = PFCEvolver(
        grid,
        pfc_params,
        dt=5e-3,
        clip=1.2,
        fft_threads=pfc_fft_threads,
        use_pyfftw=(pfc_use_pyfftw if pfc_use_pyfftw is not None else True),
    )
    solver_cfg = SolverConfig(
        dt=5e-3,
        crack_relax=crack_relax,
        history_localization_crack_threshold=localization_trigger_threshold,
        history_localization_boost=localization_trigger_boost,
        history_background_scale=localization_background_scale,
        crack_tol=crack_tol if crack_tol is not None else 1e-6,
        crack_max_iters=crack_max_iters if crack_max_iters is not None else 400,
        crack_accept_rel_residual=(
            crack_accept_rel_residual if crack_accept_rel_residual is not None else 5e-3
        ),
        crack_accept_incomplete=(
            crack_accept_incomplete if crack_accept_incomplete is not None else True
        ),
        plastic_relax=plastic_relax,
        coupling_inner_iters=max(1, int(coupling_inner_iters)),
        coupling_min_iters=max(1, int(coupling_min_iters)),
        coupling_tol=float(coupling_tol),
        coupling_stress_tol=float(coupling_stress_tol),
        post_crack_mech_correction=bool(post_crack_mech_correction),
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
    solver.initialize_state(structure.orientation, seed=int(random_seed))
    for key, value in structure.fields.items():
        solver.state[key] = value.copy()
    solver.state["history"] = np.zeros_like(structure.fields["psi"])
    solver.state["accum_plastic"] = solver.state.get("plastic", np.zeros_like(structure.fields["psi"])).copy()

    results: List[CycleResult] = []
    current_strain = 0.0
    frame_id = 0
    stress_strain_log = []
    solver_diag = {
        "step_count": 0,
        "mechanical_cg_failures": 0,
        "mechanical_runtime_warning_count": 0,
        "mechanical_nonzero_info_steps": 0,
        "mechanical_outer_not_converged_steps": 0,
        "mechanical_breakdown_steps": 0,
        "mechanical_positive_info_steps": 0,
        "mechanical_gmres_fallback_steps": 0,
        "mechanical_not_accepted_steps": 0,
        "mechanical_hold_steps": 0,
        "mechanical_solution_clipped_steps": 0,
        "mechanical_rel_residual_nonfinite_steps": 0,
        "mechanical_rel_residual_max": 0.0,
        "coupling_inner_iterations_sum": 0,
        "coupling_not_converged_steps": 0,
        "coupling_plastic_residual_max": 0.0,
        "coupling_stress_residual_max": 0.0,
        "post_crack_mech_correction_steps": 0,
        "crack_cg_nonconverged_steps": 0,
        "crack_cg_not_accepted_steps": 0,
        "nonfinite_count": 0,
        "max_abs_stress_vm": 0.0,
        "max_crack": 0.0,
        "max_abs_displacement": 0.0,
        "adaptive_split_steps": 0,
        "adaptive_split_failed_steps": 0,
    }

    # 2. 循环加载
    schedule_vals: list[float] | None = None
    if max_strain_schedule is not None:
        schedule_vals = [float(v) for v in max_strain_schedule if np.isfinite(float(v))]
        if len(schedule_vals) == 0:
            schedule_vals = None
        else:
            if cycles != len(schedule_vals):
                print(
                    f"[info] max_strain_schedule length={len(schedule_vals)} overrides cycles={cycles} "
                    f"-> cycles={len(schedule_vals)}"
                )
                cycles = len(schedule_vals)


    print_every = max(1, int(print_interval)) if print_interval is not None else 0
    vtk_every = max(1, int(vtk_interval)) if vtk_interval is not None else 0
    for cycle in range(1, cycles + 1):
        max_strain_cycle = float(schedule_vals[cycle - 1]) if schedule_vals is not None else float(max_strain)
        min_strain_cycle = -max_strain_cycle if min_strain is None else float(min_strain)
        load_segments = [max_strain_cycle] if monotonic else [max_strain_cycle, 0.0, min_strain_cycle, 0.0]
        if cycle_points is not None:
            points_per_segment = max(1, int(round(cycle_points / len(load_segments))))
            if points_per_segment * len(load_segments) != cycle_points:
                print(
                    f"[warn] cycle_points={cycle_points} not divisible by {len(load_segments)}; "
                    f"using {points_per_segment} per segment -> {points_per_segment * len(load_segments)} points"
                )
            segment_steps = points_per_segment
        print(f"=== Starting Cycle {cycle} ===")
        if schedule_vals is not None:
            print(f"  [schedule] max_strain_cycle={max_strain_cycle:.6f}")
        energy_val = 0.0
        plastic_min, plastic_max = np.inf, -np.inf
        cycle_rss_peak = -np.inf
        cycle_rss_peak_signed = 0.0
        macro = (current_strain, -poisson_ratio * current_strain, -poisson_ratio * current_strain)
        cycle_failed = False

        def _record_substep(
            step_diag: dict,
            step_idx: int,
            local_steps_now: int,
            macro_now: tuple[float, float, float],
            step_committed: bool,
        ) -> None:
            nonlocal frame_id, plastic_min, plastic_max, cycle_rss_peak, cycle_rss_peak_signed, current_strain
            solver_diag["step_count"] += 1
            solver_diag["mechanical_cg_failures"] += int(step_diag.get("mechanical_cg_failures", 0))
            solver_diag["mechanical_runtime_warning_count"] += int(
                step_diag.get("mechanical_runtime_warning_count", 0)
            )
            mech_last_info = int(step_diag.get("mechanical_last_cg_info", 0))
            if mech_last_info != 0:
                solver_diag["mechanical_nonzero_info_steps"] += 1
            if mech_last_info < 0:
                solver_diag["mechanical_breakdown_steps"] += 1
            if mech_last_info > 0:
                solver_diag["mechanical_positive_info_steps"] += 1
            if bool(step_diag.get("mechanical_gmres_fallback_used", False)):
                solver_diag["mechanical_gmres_fallback_steps"] += 1
            if str(step_diag.get("mechanical_last_solver_used", "cg")) == "hold":
                solver_diag["mechanical_hold_steps"] += 1
            if bool(step_diag.get("mechanical_solution_clipped", False)):
                solver_diag["mechanical_solution_clipped_steps"] += 1
            if not bool(step_diag.get("mechanical_last_accepted", True)):
                solver_diag["mechanical_not_accepted_steps"] += 1
            rel_res = float(step_diag.get("mechanical_last_rel_residual", 0.0))
            if not np.isfinite(rel_res):
                solver_diag["mechanical_rel_residual_nonfinite_steps"] += 1
            else:
                solver_diag["mechanical_rel_residual_max"] = max(
                    float(solver_diag["mechanical_rel_residual_max"]),
                    rel_res,
                )
            solver_diag["coupling_inner_iterations_sum"] += int(
                step_diag.get("coupling_inner_iterations_used", 0)
            )
            if not bool(step_diag.get("coupling_converged", True)):
                solver_diag["coupling_not_converged_steps"] += 1
            solver_diag["coupling_plastic_residual_max"] = max(
                float(solver_diag["coupling_plastic_residual_max"]),
                float(step_diag.get("coupling_plastic_residual", 0.0)),
            )
            solver_diag["coupling_stress_residual_max"] = max(
                float(solver_diag["coupling_stress_residual_max"]),
                float(step_diag.get("coupling_stress_residual", 0.0)),
            )
            if bool(step_diag.get("post_crack_mech_correction_used", False)):
                solver_diag["post_crack_mech_correction_steps"] += 1
            if not bool(step_diag.get("mechanical_outer_converged", True)):
                solver_diag["mechanical_outer_not_converged_steps"] += 1
            if int(step_diag.get("crack_cg_info", 0)) != 0:
                solver_diag["crack_cg_nonconverged_steps"] += 1
            if not bool(step_diag.get("crack_cg_accepted", True)):
                solver_diag["crack_cg_not_accepted_steps"] += 1
            solver_diag["nonfinite_count"] += int(step_diag.get("nonfinite_count", 0))
            solver_diag["max_abs_stress_vm"] = max(
                float(solver_diag["max_abs_stress_vm"]),
                float(step_diag.get("max_abs_stress_vm", 0.0)),
            )
            solver_diag["max_crack"] = max(
                float(solver_diag["max_crack"]),
                float(step_diag.get("max_crack", 0.0)),
            )
            solver_diag["max_abs_displacement"] = max(
                float(solver_diag["max_abs_displacement"]),
                float(step_diag.get("max_abs_displacement", 0.0)),
            )
            if not step_committed:
                return
            plast_inst = solver.state.get("plastic_inst", solver.state["plastic"])
            plast_inst_mean = plast_inst.mean()
            accum_mean = solver.state.get("accum_plastic", solver.state["plastic"]).mean()
            plastic_min = min(plastic_min, plast_inst_mean)
            plastic_max = max(plastic_max, plast_inst_mean)
            stress_tensor = solver.state["stress"]
            stress_mean = np.mean(stress_tensor, axis=(0, 1, 2))
            stress_vm_mean = float(np.mean(solver.state.get("stress_vm", 0.0)))
            rss_max, _, _, _, rss_signed_max = coupling.compute_rss(
                stress_tensor,
                backstress=solver.state.get("backstress"),
                slip_backstress=solver.state.get("chi_s"),
                slip_backstress2=solver.state.get("chi_s2"),
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

            if print_every > 0 and step_idx % print_every == 0:
                last_mech = step_diag.get("mechanical_last_solver_used", "cg")
                last_rel = step_diag.get("mechanical_last_rel_residual", 0.0)
                last_crack = step_diag.get("crack_cg_info", 0)
                print(
                    f"  Cycle {cycle} Substep {step_idx}/{local_steps_now} | Strain {current_strain:.4f} | "
                    f"mech={last_mech} rel={last_rel:.2e} crack_info={int(last_crack)}"
                )

            if vtk_every > 0 and step_idx % vtk_every == 0:
                frame_id += 1
                if vtk_dir:
                    vtk_dir.mkdir(parents=True, exist_ok=True)
                    if export_energy_fields:
                        solver.compute_energy_fields()
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
                    for key in (
                        "history",
                        "energy_elastic",
                        "energy_pfc",
                        "energy_crack",
                        "energy_total_density",
                        "crack_driving_force",
                        "toughness",
                    ):
                        value = solver.state.get(key)
                        if value is not None:
                            vtk_fields[key] = value
                    write_vtk(
                        vtk_dir / f"anim_frame_{frame_id:05d}.vtk",
                        grid,
                        vtk_fields,
                        macro_strain=macro_now,
                        deform_coordinates=True,
                    )

        for target in load_segments:
            target_start = current_strain
            target_end = target
            local_steps = int(segment_steps)
            if max_strain_step is not None:
                step_cap = float(max_strain_step)
                if np.isfinite(step_cap) and step_cap > 0.0:
                    needed = int(np.ceil(abs(target_end - target_start) / step_cap))
                    local_steps = max(local_steps, max(1, needed))
            for step in range(1, local_steps + 1):
                alpha = step / local_steps
                step_target = target_start + (target_end - target_start) * alpha
                pending_targets = [step_target]
                split_budget = max(0, int(adaptive_substep_max_splits)) if adaptive_substep_retry else 0
                while pending_targets:
                    trial_target = pending_targets.pop(0)
                    trial_increment = float(trial_target - current_strain)
                    if abs(trial_increment) <= float(adaptive_substep_min_increment):
                        solver_diag["adaptive_split_failed_steps"] += 1
                        cycle_failed = True
                        print(
                            f"[warn] cycle={cycle} step={step} rejected due to too-small increment "
                            f"({trial_increment:.3e})"
                        )
                        break

                    macro = (
                        trial_target,
                        -poisson_ratio * trial_target,
                        -poisson_ratio * trial_target,
                    )
                    energy_val = solver.step(macro)
                    step_diag = getattr(solver, "last_step_diagnostics", {})
                    step_committed = bool(
                        step_diag.get(
                            "mechanical_step_state_committed",
                            step_diag.get("mechanical_last_accepted", True),
                        )
                    )
                    if step_committed:
                        current_strain = trial_target
                    _record_substep(step_diag, step, local_steps, macro, step_committed)

                    if step_committed:
                        continue
                    if adaptive_substep_retry and split_budget > 0:
                        mid_target = current_strain + 0.5 * trial_increment
                        pending_targets.insert(0, trial_target)
                        pending_targets.insert(0, mid_target)
                        split_budget -= 1
                        solver_diag["adaptive_split_steps"] += 1
                        continue

                    solver_diag["adaptive_split_failed_steps"] += 1
                    cycle_failed = True
                    print(
                        f"[warn] cycle={cycle} step={step} rejected and split budget exhausted "
                        f"(increment={trial_increment:.3e})"
                    )
                    break

                if cycle_failed:
                    break
            if cycle_failed:
                break

        if cycle_failed:
            print(f"[STOP] Cycle {cycle} terminated due to unresolved mechanical rejects.")
            break

        crack_mean = solver.state["crack"].mean()
        accum_plastic_mean = solver.state.get("accum_plastic", solver.state["plastic"]).mean()
        crack_p95 = float(np.percentile(solver.state["crack"], 95.0))
        crack_p99 = float(np.percentile(solver.state["crack"], 99.0))
        crack_localization_index = float(crack_p99 / (crack_mean + 1e-12))
        energy_fields = solver.compute_energy_fields() if export_energy_fields else {}
        energy_elastic_mean = (
            float(np.mean(energy_fields.get("energy_elastic", 0.0)))
            if export_energy_fields
            else 0.0
        )
        energy_pfc_mean = (
            float(np.mean(energy_fields.get("energy_pfc", 0.0)))
            if export_energy_fields
            else 0.0
        )
        energy_crack_mean = (
            float(np.mean(energy_fields.get("energy_crack", 0.0)))
            if export_energy_fields
            else 0.0
        )
        energy_total_density_mean = (
            float(np.mean(energy_fields.get("energy_total_density", 0.0)))
            if export_energy_fields
            else 0.0
        )
        gnd = solver.state.get("gnd_density")
        gnd_mean = float(np.mean(gnd)) if gnd is not None else 0.0
        gnd_max = float(np.max(gnd)) if gnd is not None else 0.0
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
                gnd_mean=gnd_mean,
                gnd_max=gnd_max,
                crack_length=crack_len,
                plastic_range=plastic_range,
                rss_peak_nd=cycle_rss_peak,
                rss_peak_nd_signed=cycle_rss_peak_signed,
                crack_p95=crack_p95,
                crack_p99=crack_p99,
                crack_localization_index=crack_localization_index,
                energy_elastic_mean=energy_elastic_mean,
                energy_pfc_mean=energy_pfc_mean,
                energy_crack_mean=energy_crack_mean,
                energy_total_density_mean=energy_total_density_mean,
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
                fh.write(
                    "cycle,energy,crack_mean,accum_plastic_mean,gnd_mean,gnd_max,"
                    "crack_length,crack_delta,plastic_range,crack_p95,crack_p99,"
                    "crack_localization_index,energy_elastic_mean,energy_pfc_mean,"
                    "energy_crack_mean,energy_total_density_mean\n"
                )
                for r, pd, pr in zip(results, crack_deltas, plastic_ranges):
                    fh.write(
                        f"{r.cycle},{r.load:.6e},{r.crack_mean:.6e},"
                        f"{r.accum_plastic_mean:.6e},{r.gnd_mean:.6e},{r.gnd_max:.6e},"
                        f"{r.crack_length:.6e},{pd:.6e},{pr:.6e},"
                        f"{r.crack_p95:.6e},{r.crack_p99:.6e},{r.crack_localization_index:.6e},"
                        f"{r.energy_elastic_mean:.6e},{r.energy_pfc_mean:.6e},"
                        f"{r.energy_crack_mean:.6e},{r.energy_total_density_mean:.6e}\n"
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
            crack_loc_idx = np.array([r.crack_localization_index for r in results], dtype=float)
            energy_crack_mean = np.array([r.energy_crack_mean for r in results], dtype=float)
            energy_total_mean = np.array([r.energy_total_density_mean for r in results], dtype=float)
            with analysis_csv.open("w", encoding="utf-8") as fh:
                fh.write(
                    "cycle,a,da_dN,eps_p_half,rss_peak_nd,gnd_mean,gnd_max,"
                    "crack_localization_index,energy_crack_mean,energy_total_density_mean\n"
                )
                for c, a, dadn, eph, rp, gm, gx, cli, ecm, etm in zip(
                    cycles_out,
                    a_arr,
                    da,
                    eps_p_half,
                    rss_peak,
                    [r.gnd_mean for r in results],
                    [r.gnd_max for r in results],
                    crack_loc_idx,
                    energy_crack_mean,
                    energy_total_mean,
                ):
                    fh.write(
                        f"{int(c)},{a:.6e},{dadn:.6e},{eph:.6e},{rp:.6e},{gm:.6e},{gx:.6e},"
                        f"{cli:.6e},{ecm:.6e},{etm:.6e}\n"
                    )
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
                        np.array([sig_xx_nd, sig_yy_nd, sig_zz_nd, sig_vm_nd]),
                        stress_ref_gpa=stress_ref_gpa,
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

    if diagnostics_out is not None:
        diagnostics_out.clear()
        diagnostics_out.update(
            {
                "steps": int(solver_diag["step_count"]),
                "mechanical_cg_failures": int(solver_diag["mechanical_cg_failures"]),
                "mechanical_runtime_warning_count": int(solver_diag["mechanical_runtime_warning_count"]),
                "mechanical_nonzero_info_steps": int(solver_diag["mechanical_nonzero_info_steps"]),
                "mechanical_outer_not_converged_steps": int(solver_diag["mechanical_outer_not_converged_steps"]),
                "mechanical_breakdown_steps": int(solver_diag["mechanical_breakdown_steps"]),
                "mechanical_positive_info_steps": int(solver_diag["mechanical_positive_info_steps"]),
                "mechanical_gmres_fallback_steps": int(solver_diag["mechanical_gmres_fallback_steps"]),
                "mechanical_not_accepted_steps": int(solver_diag["mechanical_not_accepted_steps"]),
                "mechanical_hold_steps": int(solver_diag["mechanical_hold_steps"]),
                "mechanical_solution_clipped_steps": int(solver_diag["mechanical_solution_clipped_steps"]),
                "mechanical_rel_residual_nonfinite_steps": int(
                    solver_diag["mechanical_rel_residual_nonfinite_steps"]
                ),
                "mechanical_rel_residual_max": float(solver_diag["mechanical_rel_residual_max"]),
                "coupling_inner_iterations_avg": float(
                    solver_diag["coupling_inner_iterations_sum"] / max(int(solver_diag["step_count"]), 1)
                ),
                "coupling_not_converged_steps": int(solver_diag["coupling_not_converged_steps"]),
                "coupling_plastic_residual_max": float(solver_diag["coupling_plastic_residual_max"]),
                "coupling_stress_residual_max": float(solver_diag["coupling_stress_residual_max"]),
                "post_crack_mech_correction_steps": int(solver_diag["post_crack_mech_correction_steps"]),
                "crack_cg_nonconverged_steps": int(solver_diag["crack_cg_nonconverged_steps"]),
                "crack_cg_not_accepted_steps": int(solver_diag["crack_cg_not_accepted_steps"]),
                "nonfinite_count": int(solver_diag["nonfinite_count"]),
                "max_abs_stress_vm": float(solver_diag["max_abs_stress_vm"]),
                "max_crack": float(solver_diag["max_crack"]),
                "max_abs_displacement": float(solver_diag["max_abs_displacement"]),
                "adaptive_split_steps": int(solver_diag["adaptive_split_steps"]),
                "adaptive_split_failed_steps": int(solver_diag["adaptive_split_failed_steps"]),
                "stress_ref_gpa": float(stress_ref_gpa),
            }
        )

    if state_out is not None:
        state_out.clear()
        for key in (
            "crack",
            "plastic",
            "plastic_inst",
            "plastic_vec",
            "plastic_tensor",
            "accum_plastic",
            "backstress",
            "gamma_s",
            "chi_s",
            "chi_s2",
            "tau_c",
            "gnd_density",
            "displacement",
            "strain",
            "history",
            "stress",
            "stress_vm",
            "psi",
        ):
            value = solver.state.get(key)
            if value is not None:
                state_out[key] = np.array(value, copy=True)
        state_out["orientation"] = np.array(structure.orientation, copy=True)

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
