# 单晶铜拉伸/疲劳相场项目说明

## 概览
- 目标：在 3D 下耦合 PFC 密度、损伤/裂纹、等效塑性与力学平衡，模拟单晶铜（及未来多晶）在循环载荷中的裂纹萌生与疲劳规律，并输出可视化/统计用于 Paris、Coffin–Manson 对比。
- 主要变量：PFC 密度 ψ，裂纹场 φ，累积等效塑性（accum_plastic）与瞬时代理（plastic_inst）及方向分量（plastic_vec），位移场（微观校正 + 宏观总位移），应力张量。
- 能量框架：弹性正/负能分裂（谱分裂或体积-偏差分裂可切换） + 断裂能（韧性随累积塑性退化） + PFC 能 + 取向/晶界项；自由能变分得到化学势，PFC 动力学 ∂tψ = -∇²μ。

## 代码结构（关键文件）
- `sim/energy.py`：能量项、韧性退化、塑性/方向性驱动；支持谱分裂/体积-偏差分裂；`plastic_measures` 计算等效塑性与方向分量（机械 von Mises + PFC 梯度混合）。
- `sim/solver.py`：交替求解器（力学 → 塑性松弛 → 裂纹 → PFC），支持方向性驱动、应力耦合 μ_extra，跟踪 stress/stress_vm、plastic_vec。
- `sim/io.py`：LAMMPS/VTK 输出（VTK 现为二进制 STRUCTURED_GRID，可选变形坐标），输出 accum_plastic/plastic_inst/方向分量与归一化场。
- `sim/tests/virtual_cycle.py`：虚拟循环载荷驱动脚本（对称三角波），记录 CSV/标准疲劳指标 CSV、VTK、LAMMPS，拟合 Paris/Coffin–Manson 斜率。
- `report.tex`：项目报告/公式/流程/输出说明（XeLaTeX + ctex）。

## 运行示例
```bash
python -m sim.tests.virtual_cycle \
  --cycles 1 \
  --max-strain 0.08 \
  --segment-steps 50 \
  --csv-output sim/tests/virtual_cycle.csv \
  --dump-dir sim/tests \
  --vtk-dir sim/tests/virtual_cycle_vtk
```
输出：
- VTK：`sim/tests/virtual_cycle_vtk/anim_frame_*.vtk`（二进制，STRUCTURED_GRID，含 crack/accum_plastic/plastic_inst/psi/stress/displacement 等及归一化字段）。
- LAMMPS dump：`sim/tests/virtual_cycle_*.lammpstrj`（包含 crack, accum_plastic, plastic_inst, psi, plastic_vec 分量, stress_vm, stress_xx/yy/zz）。
- CSV：`sim/tests/virtual_cycle.csv`（cycle, energy, crack_mean, accum_plastic_mean, crack_length, crack_delta, plastic_range）。
- 标准疲劳指标：`sim/tests/virtual_cycle_analysis.csv`（cycle, a, da_dN, eps_p_half, rss_peak_nd）。

可视化提示：
- ParaView/Ovito 固定色标，禁用 per-timestep rescale。裂纹用 `crack_clamp03` (0–0.3) 或 `crack_norm` (0–1)；塑性/应力用 0–1；位移查看 `displacement_total`/`disp_total_norm` 或 warp by total displacement。

## COMSOL 训练/标定（可选）
本项目的力学、塑性与裂纹部分是自定义求解器，可用 COMSOL 作为“高保真参考”来做参数标定或对齐训练。推荐流程：

1) **用 COMSOL 生成参考数据**  
   - 单调/循环加载下的 **σ–ε 曲线**（平均应力、von Mises）  
   - **裂纹增长速率 / J-积分 / 能量释放率**  
   - 指定几何与边界条件下的 **应力/应变/位移场**（场数据导出）

2) **对齐/标定到本项目参数**  
   - `sim/energy.py`  
     - `CopperParameters.c11/c12/c44`：弹性常数（COMSOL 材料库或实验数据）  
     - `FractureParameters.gc/l0/epsilon_half/gres`：断裂/韧性退化参数（匹配 J-积分或裂纹增长速率）  
   - `sim/energy.py` → `PFCCoupling`  
     - `yield_tau/flow_scale/visco_exponent/linear_hardening/kin_c/kin_d`：塑性/硬化与背应力参数（匹配循环滞回或屈服平台）  
   - `sim/solver.py` → `SolverConfig`  
     - `plastic_relax/dir_coupling/stress_mu_weight`：宏观加载下的软化与方向性耦合强度  

3) **接口实现（COMSOL Bridge）**  
   - 见 `comsol_bridge/README.md`：在 Windows 端启动桥接服务（连接 COMSOL Server），本地 Python 调接口提交任务、取回结果。  
   - 模型建议预先配置好 Export 节点（表格/曲线/场），任务只负责传参、求解、导出。  

> 若需要，我可以根据你已有的 COMSOL 模型/导出节点，直接生成 “参数扫描 + 结果回传 + 拟合” 的 Python 脚本模板。

## 最近更新
- 塑性/方向场：新增 `plastic_measures`，机械 von Mises/轴向应变与 PFC 梯度按权重混合（默认机械占比 0.9），输出塑性向量。
- 方向性裂纹驱动：加载轴塑性分量放大历史能量/驱动力（`dir_coupling` 默认 0.8）。
- 应力耦合：von Mises 应力归一化后加入 μ_extra，PFC 在高应力区更敏感。
- 循环载荷：对称三角波 0→+ε_max→0→−ε_max→0，每段 50 步，支持失败阈值提前停，记录每周期 crack/plastic 的均值与增量/范围。
- 单向分裂可切换：谱分裂/体积-偏差分裂可选，并在回归中比较 Mode-I/压缩 φ 演化。
- PFC 更新：引入线性半隐式 FFT 步（默认），显著放宽 dt 稳定性。
- 输出：VTK 改为二进制 STRUCTURED_GRID，附归一化场；LAMMPS/VTK 统一使用 accum_plastic 与 plastic_inst。
- 报告：`report.tex` 同步上述公式/流程/输出说明，添加当前结果快照。

## 当前结果快照（默认参数：128×64×16，缺陷幅值 0.12，ε_max=0.08，dir_coupling=0.8）
- 裂纹：最高 ~0.034，轻微均匀损伤，未贯穿。
- 塑性：标量 ~0.7–0.9 累积，方向分量卸载后接近 0。
- 应力/位移：加载时宏观梯度明显；卸载到 0 时应力/位移回零，数值稳定。
- 可视化：二进制 VTK 无读错，固定色标可避免“全红”假象。

## 待拓展
- 增载或延长循环/调韧性与缺陷以触发裂纹局部化；输出能量密度场（弹性能/断裂能）用于裂尖能量分析。
- 引入多晶/晶界取向场与非周期边界（FFT/FEM 替换 CG）。

## 新增/更新内容（缺陷播种与快速演化验证）
- 新增 `sim/defects.py`：支持缺陷播种配置（密度、区域、类型概率）、权重场引导播种、折线位错（确定性线段离散）以及各向异性核；输出缺陷掩膜 `defect_mask`/`line_mask`，将播种点平滑成 ψ/裂纹/塑性初值。
- `Cu111StructureBuilder` 接受 `defect_config`：可用上面的播种器替代旧的随机掩膜；仍兼容原噪声/掩膜流程。
- 示例脚本 `scripts/generate_seeded_cu.py`：可生成带播种缺陷的 Cu 初值，输出 LAMMPS data/dump 和 VTK（含 `defect_mask` 等），参数包括区域、密度、类型概率、核宽度等。
- 冒烟演化脚本 `sim/tests/smoke_seeded.py`：初始化播种缺陷后跑少量步长，逐步输出 VTK 序列（含变形坐标）和末步 LAMMPS，便于 ParaView 时间序列查看。
- 变形坐标：VTK 支持 `deform_coordinates=True`，将宏观应变+微观位移写入坐标；也会输出未变形版本便于对比。
- 疲劳主脚本 `sim/tests/virtual_cycle.py` 增强：支持缺陷播种输入、缺口种子（notch_box）、预演化步（pre_relax）让缺陷/滑移带先成形；宏观应变包含泊松收缩 (ε, -ν ε, -ν ε)；可选应力耦合到 PFC (`stress_mu_weight`)；韧性缩放 `toughness_scale` 便于促开裂/稳健性调参；逐步 VTK/LAMMPS 输出使用一致的宏观应变。
- 塑性/力学更新：`plastic_measures` 采用 FCC 正交滑移系 RSS（含符号）和屈服阈值/流动尺度，累积等效塑性（用于韧性退化）与塑性张量（本征应变）。力学求解与能量使用 $(\varepsilon - \varepsilon^p)$；支持晶界弱化 mask。输出可选应力–应变曲线 CSV（`stress_strain_csv`）。
- 输出宏观应力应变：`virtual_cycle.py` 可选写出逐步的平均应力-应变曲线（含 von Mises）到 CSV（`stress_strain_csv`），便于快速绘制。
- 诊断：`sim/energy.py` 增加裂纹驱动力/能量一致性的一次性诊断接口（`diagnose_crack_energy_consistency` / `diagnose_phi_consistency`）。

## 回归测试（自动化）
新增三类回归脚本，用于验证边界处理与裂纹驱动力：

- 小规模快速回归：`sim/tests/regress_bc_crack.py`
- 大网格多周期回归：`sim/tests/regress_bc_crack_large.py`
- 微米尺度回归：`sim/tests/regress_bc_crack_micron.py`（spacing=1e-6，gc/l0 按 spacing 缩放以保持能量比例）

### 运行方式
```bash
python sim/tests/regress_bc_crack.py
python sim/tests/regress_bc_crack_large.py
python sim/tests/regress_bc_crack_micron.py
```

### 统一入口（推荐）
```bash
python sim/tests/regress_all.py --strict --task boundary_crack
```

默认日志位置（自动生成）：`sim/tests/regress_runs/YYYY-MM-DD/<task>/`  
其中包含：`summary.json` 与每个子测试的 `*.json`/`*.stdout`/`*.stderr`。

### 其他测试输出目录（统一命名）
除回归外，其他测试输出默认写入：`sim/tests/runs/YYYY-MM-DD/<task>_<HHMMSS>/`  
例如：
- `virtual_cycle`：`sim/tests/runs/2026-02-02/virtual_cycle_153012/`
- `seeded_cu_smoke`：`sim/tests/runs/2026-02-02/seeded_cu_smoke_153745/`

### 严格阈值 + JSON 输出
```bash
python sim/tests/regress_bc_crack.py --strict --output /tmp/regress_small.json
python sim/tests/regress_bc_crack_large.py --strict --output /tmp/regress_large.json
python sim/tests/regress_bc_crack_micron.py --strict --output /tmp/regress_micron.json
```

### 输出内容
每个脚本都会输出 JSON，包含：
- `results`：核心指标（压缩裂纹增长、补丁残差、Mode-I 裂纹增长等）
- `thresholds`：判定阈值
- `timing`：每个子测试与总耗时（秒）
- `passed` / `failures`
 - `split_compare`：谱分裂 vs 体积-偏差分裂的 Mode-I/压缩对比

> 说明：大网格脚本采用 **三向压缩**（避免单向压缩引入局部拉应变），Mode‑I 脚本包含多周期加载。

### 验证记录（2026-02-02）
- 回归（strict）：`python sim/tests/regress_all.py --strict --log-dir sim/tests/regress_runs/2026-02-02/boundary_crack`  
  结果：small/large/micron 全通过，谱分裂优于体积‑偏差分裂。日志：`sim/tests/regress_runs/2026-02-02/boundary_crack/summary.json`  
- 多物理快速检查（单调拉伸，1 周期，10 步/段，max_strain=0.01）：  
  `crack_mean=0`, `crack_length=0`, `accum_plastic_mean=2.895e-04`, `plastic_range=1.710e-02`, `rss_peak_nd=4.751e-03`  
  注：1 周期下 Coffin–Manson 拟合为弱约束（polyfit 提示条件数不足）。
- 多物理循环检查（2 周期，15 步/段，max_strain=0.02）：  
  结果：`crack_mean=0`, `crack_length=0`, `accum_plastic_mean=3.634e-03`, `plastic_range=1.877e-03`, `rss_peak_nd=9.980e-03`  
  输出：`sim/tests/virtual_cycle_long.csv`, `sim/tests/virtual_cycle_long_analysis.csv`, `sim/tests/virtual_cycle_long_stress_strain.csv`,  
  应力–应变曲线图：`sim/tests/virtual_cycle_long_stress_strain.png`  
  曲线指标：`sig_xx_GPa` ∈ [−3.149, 3.063]，`sig_vm_GPa(max)=4.476`，小应变等效模量 `E_eff≈153.16 GPa`（|ε|≤0.002 线性拟合）
- 多物理循环检查（10 周期，100 步/段，max_strain=0.02）：  
  结果：`crack_mean=0`, `crack_length=0`, `accum_plastic_mean=1.216e-01`, `plastic_range=3.676e-03`, `rss_peak_nd=9.519e-03`  
  输出：`sim/tests/runs/2026-02-02/virtual_cycle_10c_100s_212301/virtual_cycle.csv`,  
  标准疲劳指标：`sim/tests/runs/2026-02-02/virtual_cycle_10c_100s_212301/virtual_cycle_analysis.csv`，  
  应力–应变曲线图：`sim/tests/runs/2026-02-02/virtual_cycle_10c_100s_212301/virtual_cycle_stress_strain.png`  
  曲线指标：`sig_xx_GPa` ∈ [−3.237, 3.049]，`sig_vm_GPa(max)=4.594`，小应变等效模量 `E_eff≈135.99 GPa`（|ε|≤0.002 线性拟合）  
  注：裂纹长度未增长，Paris/Coffin–Manson 拟合为弱约束（da/dN≈0）。
