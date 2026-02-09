# 单晶铜拉伸/疲劳相场项目说明

## 概览
- 目标：在 3D 下耦合 PFC 密度、损伤/裂纹、等效塑性与力学平衡，模拟单晶铜（及未来多晶）在循环载荷中的裂纹萌生与疲劳规律，并输出可视化/统计用于 Paris、Coffin–Manson 对比。
- 主要变量：PFC 密度 ψ，裂纹场 φ，累积等效塑性（accum_plastic）与瞬时代理（plastic_inst）及方向分量（plastic_vec），位移场（微观校正 + 宏观总位移），应力张量。
- 能量框架：弹性正/负能分裂（谱分裂或体积-偏差分裂可切换） + 断裂能（韧性随累积塑性退化） + PFC 能 + 取向/晶界项；自由能变分得到化学势，PFC 动力学 ∂tψ = -∇²μ。

## 环境准备
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 可选开发工具
pip install -r requirements-dev.txt
```

## 代码结构（关键文件）
- `sim/energy.py`：能量项、韧性退化、塑性/方向性驱动；支持谱分裂/体积-偏差分裂；`plastic_measures` 计算等效塑性与方向分量（机械 von Mises + PFC 梯度混合）。
- `sim/solver.py`：交替求解器（力学 → 塑性松弛 → 裂纹 → PFC），支持方向性驱动、应力耦合 μ_extra，跟踪 stress/stress_vm、plastic_vec。
- `sim/io.py`：LAMMPS/VTK 输出（VTK 现为二进制 STRUCTURED_GRID，可选变形坐标），输出 accum_plastic/plastic_inst/方向分量与归一化场。
- `sim/tests/virtual_cycle.py`：虚拟循环载荷驱动脚本（对称三角波），记录 CSV/标准疲劳指标 CSV、VTK、LAMMPS，拟合 Paris/Coffin–Manson 斜率。
- `sim/tests/regress_microstrain.py`：微应变线弹性回归（σ–ε 比值与塑性漂移）。
- `sim/tests/regress_gnd.py`：GND/Nye 回归（滑移梯度驱动下 ρ_GND 线性响应检查）。
- `sim/tests/regress_gnd_cycle.py`：低幅循环 GND 增长回归（输出 gnd_density_mean/Σ|γ_s| 趋势与参数快照）。
- `sim/tests/plot_gnd_cycle.py`：gnd_cycle.csv 基线可视化（GND mean 与 accum_plastic 同图）。
- `sim/tests/scan_gnd_orientations.py`：扫描晶向对 GND 分布/统计的敏感性。
- `sim/tests/scan_hgnd_cycle.py`：h_gnd 4 点扫描（低幅循环），自动输出对比图/summary.csv。
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

## Phase-1 工作流（本周）
配置化运行：
```bash
python sim/tests/run_virtual_cycle_config.py --config sim/configs/monotonic_baseline.yaml --dry-run
python sim/tests/run_virtual_cycle_config.py --config sim/configs/fatigue_lowamp.yaml
python sim/tests/run_virtual_cycle_config.py --config sim/configs/notch_gnd.yaml
python sim/tests/run_virtual_cycle_config.py --config sim/configs/monotonic_baseline.yaml --summary-output sim/tests/runs/YYYY-MM-DD/phase1_config_runs/monotonic_baseline_run_summary.json
```

一键回归套件：
```bash
python sim/tests/run_phase1_suite.py --strict --out sim/tests/regress_runs/YYYY-MM-DD/phase1_suite
```
输出：`summary.json` + 每项 `*.stdout`/`*.stderr` + 各子测试 JSON。

## Phase-2 工作流（稳定性 + 裂纹萌生）
单配置稳定性检查（带阈值）：
```bash
python sim/tests/run_virtual_cycle_config.py \
  --config sim/configs/monotonic_baseline.yaml \
  --max-runtime-warnings 20 \
  --max-mechanical-not-accepted-steps 0 \
  --max-crack-cg-nonconverged-steps 0 \
  --max-nonfinite-count 0
```

裂纹萌生扫描：
```bash
python sim/tests/scan_crack_onset.py \
  --config sim/configs/crack_onset_scan.yaml \
  --no-auto-output \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 20 \
  --max-nonfinite-count 0
```

Phase-2 门禁（推荐）：
```bash
python sim/tests/regress_phase2.py \
  --strict \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 20 \
  --scan-config sim/configs/crack_onset_scan.yaml
```

Phase-2 + 实验对齐一体门禁（当前主推荐）：
```bash
python sim/tests/regress_phase2.py \
  --strict \
  --scan-config sim/configs/crack_onset_scan.yaml \
  --with-exp-alignment \
  --exp-alignment-config sim/configs/fatigue_lowamp_align_locked_v4.yaml \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 20 \
  --max-nonfinite-count 0
```

快速烟测（小网格，开发时）：
```bash
python sim/tests/regress_phase2.py \
  --skip-phase1-suite \
  --scan-config sim/configs/crack_onset_scan_quick.yaml \
  --scan-max-cases 1 \
  --scan-min-onset-cases 0 \
  --max-runtime-warnings 200
```

说明：
- `run_virtual_cycle_config.py` 现会输出 `runtime_warning_count`、`runtime_warning_items` 与 `stability_diagnostics`（含 mechanical/crack CG 与 nonfinite 计数）。
- `scan_crack_onset.py` 会输出 `summary.json` + `summary.csv`，并区分 `onset_length`（长度主判据）与 `onset_mean_aux`（均值辅助判据）。
- `scan_crack_onset.py` 支持 notch 轨迹门禁 `min_notch_cycles_completed`（YAML `criteria` 或 CLI `--min-notch-cycles-completed`），并在 case 结果中输出 `cycles_ok`/`notch_case`。
- `scan_crack_onset.py` 支持 `--no-auto-output`，可关闭每 case 的 VTK/LAMMPS 落盘用于加速筛选。
- Week-3 参数 DOE 可用 `sim/tests/sweep_crack_onset_doe.py`：
```bash
python sim/tests/sweep_crack_onset_doe.py \
  --base-config sim/configs/crack_onset_scan_quick.yaml \
  --tag doe_week3_quick \
  --max-runs 4 \
  --max-cases 1 \
  --min-notch-cycles-completed 1 \
  --scan-timeout-s 180 \
  --mech-regularization-values 1.0,2.0 \
  --mech-solution-abs-limit-values 8,10 \
  --mech-accept-rel-residual-values 0.008
```
  输出：`runs.csv`（组合排序）+ `cases.csv`（逐 case 指标）+ `summary.json`（top-runs）。
  说明：可用 `--vc-cycles/--vc-cycle-points/--vc-mech-*` 建立“数值预筛层”，先比较稳定性再回到 full scan。
- `sim/configs/crack_onset_scan.yaml` 默认使用 `crack_length_threshold=0.995` + `failure_threshold=0.999`，用于恢复 length-led onset 判据并支持 post-onset 轨迹对齐。
- `sim/configs/crack_onset_scan.yaml` 现锁定 `crack_max_iters=1200`，用于抑制 `notch_medium_drive` 的 crack-CG 非收敛高频问题。
- 默认机械策略切到 `volumetric + jacobi`，并启用 `mech_clip_solution_on_limit=true`（解超限裁剪而非直接拒收）。
- 标定闭环执行模板见 `docs/calibration_phase2.md`。
- Week-4 对齐参数小扫描工具：`sim/tests/sweep_exp_alignment.py`。
- Week-4 第一轮对齐报告：`docs/week4_alignment_round1_2026-02-07.md`。
- Week-4 seed 稳健性复验工具：`sim/tests/repeat_crack_onset_seeds.py`。
- Week-4 seed 稳健性报告：`docs/week4_seed_robustness_round1_2026-02-07.md`。
- Week-4 发布基线一键入口：`sim/tests/run_release_baseline_week4.py`。
- Week-4 发布基线说明：`docs/week4_release_baseline_pack_2026-02-07.md`。
- Week-8 多工况对齐配置：`sim/configs/exp_alignment_multi_week8.yaml`（5 个循环工况，默认复用首工况 `sim_csv` 以加速）。
- `sim/tests/regress_exp_alignment_multi.py` 支持 `--reuse-first-sim-csv`（或在 YAML `defaults.reuse_first_sim_csv=true`）用于多工况快速门禁。
- Week-8 标定扫描骨架：`sim/tests/sweep_calibration_multi.py`（候选参数网格扫描、排名、最佳配置导出、lock 草案写出）。
- Week-8 20-seed 分批模板：`sim/tests/run_seed_robustness_20.py`（默认 `41-60`，每批 5 个，并自动产出 CI 汇总）。
- Week-8 seed 置信区间统计：`sim/tests/summarize_seed_robustness_ci.py`（支持跨批次聚合 + Wilson CI）。
- Week-8 对齐汇总：`docs/week8_multi_condition_summary_2026-02-08.md`。
- Week-8 seed/产物管理说明：`docs/week8_seed_ci_and_artifact_rules_2026-02-08.md`。
- Week-9 产物分层治理规则：`docs/week9_artifact_governance_2026-02-08.md`。
- Week-9 产物清理脚本：`sim/tests/cleanup_artifacts.sh`（默认 dry-run，`--apply` 实删）。
- D1 全量门禁入口（非 quick，全 case）：`sim/tests/run_d1_full_gate.py`。
- D1 一键命令包：`sim/tests/d1_full_gate_commands.sh`（默认包含 full-case seed 稳健性批次）。
- D1 多工况真实配置（5 条件）：`sim/configs/exp_alignment_multi_d1_full.yaml`。
- Week-8 任务清单：`WEEK8_CHECKLIST.md`。
- 本周任务跟踪见 `WEEKLY_CHECKLIST.md`。
- 2026-02-07 全量一体门禁通过记录：
  `sim/tests/regress_runs/2026-02-07/phase2_gate_with_exp_full_locked/summary.json`（`passed=true`）。

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

> 若需要，可以根据已有的 COMSOL 模型/导出节点，直接生成 “参数扫描 + 结果回传 + 拟合” 的 Python 脚本模板。

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
- 示例脚本 `sim/tests/build_cu111.py`：可生成带缺陷初值的 Cu 数据，输出 LAMMPS data/dump 供可视化检查。
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
python sim/tests/regress_all.py --strict --log-dir sim/tests/regress_runs/YYYY-MM-DD/boundary_crack
```

日志目录通过 `--log-dir` 显式指定。建议统一写到：`sim/tests/regress_runs/YYYY-MM-DD/<task>/`。  
其中包含：`small.json`/`large.json`/`micron.json`、`*.stdout`/`*.stderr` 与汇总输出。

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

### 验证记录（2026-02-05）
- 循环稳定判据：`N=5`、`tol=2%`（指标：`plastic_range`、`rss_peak_nd`）。
- 实验对比（第 1000 周期，轴 -1,1,1，(111)[-101]，微应变×1e-6）：  
  修正 Schmid：`m=0.272166`；实验/模拟叠加图：`sim/tests/runs/2026-02-05/exp_compare_cycle_00001000_151235/schmid_overlay.png`  
  实验修正 CSV：`sim/tests/runs/2026-02-05/exp_compare_cycle_00001000_151235/experiment_schmid.csv`  
  模拟修正 CSV：`sim/tests/runs/2026-02-05/exp_compare_cycle_00001000_151235/simulation_schmid.csv`
- 稳定循环模拟（200 点/周期，max_strain=0.001286103，orientation=-1,1,1）：  
  `cycles_run=9`，稳定后停止；`crack_mean=0`, `crack_length=0`, `accum_plastic_mean=2.683e-02`, `plastic_range=7.082e-04`, `rss_peak_nd=2.287e-04`  
  输出：`sim/tests/runs/2026-02-05/fatigue_exp_match_200pts_tuned_fast_131238/virtual_cycle.csv`，  
  `sim/tests/runs/2026-02-05/fatigue_exp_match_200pts_tuned_fast_131238/virtual_cycle_stress_strain.csv`
  说明：加速版机械求解参数（`mech_max_iters=120`, `mech_outer_max_iters=3`, `mech_tol=2e-5`, `mech_outer_tol=2e-6`）。
- 回归（2026-02-05）：  
  微应变线弹性：`sim/tests/regress_runs/2026-02-05/microstrain/summary.json`（passed）  
  边界裂纹（小/默认）：`sim/tests/regress_runs/2026-02-05/bc_crack/summary.json`（passed）  
  边界裂纹（large）：`sim/tests/regress_runs/2026-02-05/bc_crack_large/summary.json`（passed）  
  边界裂纹（micron）：`sim/tests/regress_runs/2026-02-05/bc_crack_micron/summary.json`（passed）  
  GND/Nye：`sim/tests/regress_runs/2026-02-05/gnd/summary.json`（passed）  
  低幅循环 GND：`sim/tests/regress_runs/2026-02-05/gnd_cycle/summary.json`（passed）  
  基线图：`sim/tests/regress_runs/2026-02-05/gnd_cycle/gnd_cycle_baseline.png`  
  h_gnd 对比：  
  - h_gnd=0：`sim/tests/regress_runs/2026-02-05/gnd_cycle_hgnd0/summary.json`  
  - h_gnd=1e-4：`sim/tests/regress_runs/2026-02-05/gnd_cycle_hgnd1e-4/summary.json`  
  对比图：`sim/tests/regress_runs/2026-02-05/gnd_cycle_compare.png`
- h_gnd 扫描（4 点，低幅循环）：  
  根目录：`sim/tests/regress_runs/2026-02-05/gnd_cycle_hgnd_scan_4pt/`  
  总结：`summary.csv`，对比图：`gnd_cycle_hgnd_scan.png`
- 晶向扫描（低幅循环 + notch，小网格）：  
  输出根目录：`sim/tests/runs/2026-02-05/gnd_orient_scan_lowamp_notch_small_180348/`  
  方向：`[100]`, `[110]`, `[111]`, `[112]`；summary.json 含 gnd_mean/gnd_max/accum_plastic。
- 晶向 × h_gnd 灵敏度（低幅循环 + notch，cycles=5，小网格）：  
  输出根目录：`sim/tests/runs/2026-02-05/gnd_orient_hgnd_sens_193218/`  
  汇总：`summary.csv`；对比图：`gnd_mean_vs_hgnd.png` / `accum_plastic_vs_hgnd.png`

### 验证记录（2026-02-06，Phase-1 收口）
- Fresh venv 验证：新建 `.venv_phase1_check` 后安装 `requirements.txt`，并执行  
  `python sim/tests/run_phase1_suite.py --strict --out sim/tests/regress_runs/2026-02-06/phase1_suite_venvcheck`，  
  结果 `passed=true`，汇总：`sim/tests/regress_runs/2026-02-06/phase1_suite_venvcheck/summary.json`。
- 配置化实跑结果：
  `monotonic_baseline`（1 周期）、`fatigue_lowamp`（稳定停止于第 5 周期）、
  `notch_gnd`（完成 8 周期），对应汇总分别为  
  `sim/tests/runs/2026-02-06/phase1_config_runs/monotonic_baseline_run_summary.json`、  
  `sim/tests/runs/2026-02-06/phase1_config_runs/fatigue_lowamp_run_summary.json`、  
  `sim/tests/runs/2026-02-06/phase1_config_runs/notch_gnd_run_summary.json`。
- 物理映射冻结（Phase-1 统一约定）：
  `L0=1e-6 m`、`sigma_ref=168.4 GPa`、`b_phys=2.556e-10 m`、推荐 `gnd_burgers_nd=2.556e-4`，  
  详见 `docs/units_mapping.md` 与 `docs/parameter_register.md`。
- 数值备注：`fatigue_lowamp` 与 `notch_gnd` 运行时有 `RuntimeWarning`（CG/梯度运算），
  但周期汇总指标为有限值，已在 `HANDOFF.md` 标记为后续数值稳定性事项。

### 验证记录（2026-02-06，Phase-2 快速烟测）
- 稳定性单配置检查（monotonic）：
  `sim/tests/regress_runs/2026-02-06/monotonic_stability_check_summary.json`  
  结果：`passed=true`，`runtime_warning_count=0`，`nonfinite_count=0`。
- 裂纹萌生扫描快速配置：
  `python sim/tests/scan_crack_onset.py --config sim/configs/crack_onset_scan_quick.yaml`  
  输出：`sim/tests/regress_runs/2026-02-06/crack_onset_scan_quick_smoke/summary.json`（passed）。
- Phase-2 快速门禁（跳过完整 Phase-1 套件）：
  `sim/tests/regress_runs/2026-02-06/phase2_gate_quickcheck/summary.json`  
  结果：`passed=true`。

### 验证记录（2026-02-06，Phase-2 阈值锁定）
- 全量裂纹萌生扫描（4 cases）：
  `sim/tests/regress_runs/2026-02-06/crack_onset_scan_full_locked/summary.json`  
  结果：`passed=true`，`onset_cases=3/4`，`runtime_warning_count=0`（全部 case）。
- 锁定扫描阈值（`sim/configs/crack_onset_scan.yaml`）：
  - `min_onset_cases=1`
  - `max_runtime_warnings=50`
  - `min_crack_mean_delta=5.0e-4`
- 机械/裂纹求解稳定性策略（本轮）：
  - 机械：CG 残差验收 + 正则 + 解幅值上限 + 可选 GMRES 回退。
  - 裂纹：允许接受有限的非零 `crack_cg_info` 结果，避免裂纹更新冻结。

### 验证记录（2026-02-06，Phase-2 判据收紧）
- 全量裂纹萌生扫描（4 cases）：
  `sim/tests/regress_runs/2026-02-06/crack_onset_scan_full_locked_v3/summary.json`  
  结果：`passed=true`，`onset_cases=3/4`，`runtime_warning_count=0`（全部 case）。
- 判据更新（`sim/configs/crack_onset_scan.yaml`）：
  - `min_crack_delta=5.0e-2`（长度主判据）
  - `allow_mean_aux=true` + `min_crack_length_for_mean_aux=1.0e-1`
  - `max_mechanical_not_accepted_steps=160`
  - `max_crack_cg_nonconverged_steps=80`
  - `max_runtime_warnings=50`
- 结果解读：
  - notch 三个 case：在 `reg=1e-4` 基线下 `onset_length=true`；在新 `reg=1.0` 配置下更偏 `onset_mean_aux=true`（长度很快饱和）。
  - no-notch 对照：`onset=false` 且 `checks_ok=true`（作为负对照保留）

### 验证记录（2026-02-06，unilateral 分支预条件/算子替换）
- 对照矩阵（原策略，`limit=10`）：
  `sim/tests/regress_runs/2026-02-06/unilateral_matrix_l10/`  
  结果：`spectral/volumetric × none/jacobi` 四组均为 `mechanical_not_accepted_steps=320`。
- 采用新策略（`volumetric + jacobi + mech_clip_solution_on_limit=true + mech_regularization=1e-4`）后：
  - 单 case 320-step：`sim/tests/regress_runs/2026-02-06/crack_onset_aggressive_clip_320/summary.json`
    - `mech_not_accepted_steps=0`
  - 全扫描 4 cases：`sim/tests/regress_runs/2026-02-06/crack_onset_scan_clip_full/summary.json`
    - 四个 case 均 `mech_not_accepted_steps=0`，满足 `<160` 目标。
- 备注：notch case 中 `mechanical_solution_clipped_steps` 较高，后续应继续压低裁剪依赖（通过更稳健线性算子/预条件组合）。

### 验证记录（2026-02-07，降裁剪率 + 裂纹 CG 阈值收紧）
- 机械裁剪率专项：
  - 代理筛选表明 `mech_regularization=1.0` 才能明显降低裁剪依赖（同样 `limit=10`、`clip=true` 条件下）。
  - 正式全扫描：`sim/tests/regress_runs/2026-02-07/crack_onset_scan_full_v5_reg1/summary.json`
    - notch 三 case：`mechanical_solution_clipped_steps=118/117/120`（对应 `steps=160`）
    - `mechanical_not_accepted_steps=0`（全部 case，保持 `<160`）
- 裂纹 CG 阈值收紧：
  - 将 `max_crack_cg_nonconverged_steps` 从 `320` 收紧到 `80`。
  - 同一全扫描中 notch 三 case 分别为 `52/42/40`，满足收紧阈值。

### 验证记录（2026-02-07，reg=2.0 + length-led 回正，首轮）
- 全量裂纹萌生扫描（4 cases，`cg<=20`）：
  `sim/tests/regress_runs/2026-02-07/crack_onset_scan_reg2_len0995_fail0995_cg20_full/summary.json`
- 参数落点（首轮）：
  - `mech_regularization=2.0`
  - `crack_length_threshold=0.995`
  - `failure_threshold=0.995`
  - `max_crack_cg_nonconverged_steps=20`
- 关键结果（notch 三 case）：
  - `mechanical_solution_clipped_steps=22/23/21`（目标 `<80/160` 达成）
  - `mechanical_not_accepted_steps=0`（保持 `<160`）
  - `crack_cg_nonconverged_steps=7/6/7`（`<=20` 达成）
  - `onset_length=true` 且 `onset_mean_aux=true`（长度主判据恢复）
- Phase-2 对齐回归：
  `sim/tests/regress_runs/2026-02-07/phase2_gate_reg2_len0995_fail0995_cg20/summary.json`（`passed=true`）。

### 验证记录（2026-02-07，post-onset 轨迹对齐第 2 轮）
- 当前锁定参数（配置默认）：
  - `mech_regularization=2.0`
  - `crack_length_threshold=0.995`
  - `failure_threshold=0.999`
  - `max_crack_cg_nonconverged_steps=20`
- `failure_threshold` 扫描（notch 3 case）：
  - `fail=0.998`：`sim/tests/regress_runs/2026-02-07/crack_onset_scan_reg2_len0995_fail0.998_n3/summary.json`
    - mild/medium 可达 4 周期，aggressive 仍 2 周期。
  - `fail=0.999`：`sim/tests/regress_runs/2026-02-07/crack_onset_scan_reg2_len0995_fail0.999_n3/summary.json`
    - 三个 notch case 均达 4 周期。
- 全量 4 case 复核（`cg<=20`）：
  `sim/tests/regress_runs/2026-02-07/crack_onset_scan_reg2_len0995_fail0999_cg20_full/summary.json`
  - notch 三 case：`cycles_completed=4`、`onset_length=true`
  - `mechanical_solution_clipped_steps=22/25/25`（`steps=320`）
  - `mechanical_not_accepted_steps=0`
  - `crack_cg_nonconverged_steps=7/6/7`
- Phase-2 对齐回归（fail=0.999）：
  `sim/tests/regress_runs/2026-02-07/phase2_gate_reg2_len0995_fail0999_cg20/summary.json`（`passed=true`）。

### 验证记录（2026-02-08，Week-8 多工况 + 能量门禁 + CI）
- 多工况对齐 smoke（5 工况）：
  `sim/tests/regress_runs/2026-02-08/exp_alignment_multi_week8_smoke/summary.json`（`passed=true`，`passed_count=5/5`）。
  条件明细见 `sim/tests/regress_runs/2026-02-08/exp_alignment_multi_week8_smoke/conditions/*.summary.json`，
  汇总均值：`avg_rmse_tau≈27.776 MPa`、`avg_mae_tau≈22.723 MPa`、`avg_rmse_gamma≈0.003808`。
- 能量一致性门禁 smoke：
  `sim/tests/regress_runs/2026-02-08/energy_gate_smoke/summary.json`（`passed=true`）；
  `n_cycles=5`、`energy_drop_count=0`、`crack_reversal_count=0`、`plastic_reversal_count=0`。
- Phase-2 + exp + energy 一体 smoke：
  `sim/tests/regress_runs/2026-02-08/phase2_with_energy_gate_smoke/summary.json`（`passed=true`）。
- CI smoke（本地复核）：
  `sim/tests/regress_runs/2026-02-08/ci_smoke_local_verify_fix/summary.json`（`passed=true`，`phase2_quick/multi_align_smoke/seed_ci_smoke` 全通过）。
- Week-4 release（含 seed 批次）：
  `sim/tests/regress_runs/2026-02-08/release_baseline_week4_fullskip_with_seeds/bundle_summary.json`（`passed=true`，`phase2_gate + seed_batch_1 + seed_batch_2` 全通过）。

### 验证记录（2026-02-09，D1 全量门禁）
- D1 full gate（非 quick、全 case、seed full-mode）：
  `sim/tests/regress_runs/2026-02-09/d1_full_gate/summary.json`（`passed=true`，`failure_reasons=[]`）。
- 验收矩阵：
  - `acceptance.phase2_full.passed=true`
  - `acceptance.multi_align_full.passed=true`（`condition_total=5`，`passed_count=5`）
  - `acceptance.seed_robustness.passed=true`（`seed_gate_pass_count=6/6`）

### 验证记录（2026-02-09，D2 裂纹局部化触发 + 能量密度输出）
- D2 门禁：
  `sim/tests/regress_runs/2026-02-09/d2_localization_energy/summary.json`（`passed=true`）。
- 核心指标：
  - `cycles_completed=6`
  - `crack_delta_total=2.396875`（阈值 `>=0.05`）
  - `crack_localization_index_peak=5.1072`（阈值 `>=3.0`）
  - `energy_crack_mean_final=0.1322`
  - `energy_total_density_mean_final=1.2391`
  - `vtk_energy_field_count=6`（`energy_elastic/energy_pfc/energy_crack/energy_total_density/crack_driving_force/toughness`）

### 验证记录（2026-02-09，D3 多物理耦合矩阵）
- D3 矩阵 smoke：
  `sim/tests/regress_runs/2026-02-09/d3_multiphysics_matrix_smoke/summary.json`（`passed=true`，`case_total=3`，`passed_count=3`）。
- 矩阵覆盖：
  - 正向 case：`notch_positive`（高裂纹增量 + 局部化）
  - 正向强裂纹 case：`notch_strong_positive`（高局部化 + `crack_mean_final` 下限）
  - 负对照：`no_notch_negative`（抑制误触发，`crack_mean_final` 与能量密度保持低值）

### 验证记录（2026-02-09，release_d2_d3_quick_seedfull 完整包）
- 汇总：
  `sim/tests/regress_runs/2026-02-09/release_d2_d3_quick_seedfull/bundle_summary.json`（`passed=true`）。
- 通过矩阵：
  - `acceptance.d2_localization.passed=true`
  - `acceptance.d3_matrix.passed=true`（`case_total=3`，`passed_count=3`）
  - `seed_batch_1`: `all_seed_gate_passed=true`（`3/3`）
  - `seed_batch_2`: `all_seed_gate_passed=true`（`3/3`）
- 同日快速基线（不含 seed）：
  `sim/tests/regress_runs/2026-02-09/release_d2_d3_quick_baseline/bundle_summary.json`（`passed=true`）。

## 基准测试全景矩阵（截至 2026-02-09）

| 基准脚本/入口 | 验证目标 | 最新通过产物 | 核心结果 |
|---|---|---|---|
| `sim/tests/regress_microstrain.py` | 线弹性微应变比值 + 低塑性漂移 | `sim/tests/regress_runs/2026-02-05/microstrain/summary.json` | `passed=true`，`stress_ratio≈2.0` |
| `sim/tests/regress_gnd.py` | GND 线性响应（双梯度比值） | `sim/tests/regress_runs/2026-02-05/gnd/summary.json` | `passed=true`，`ratio_double≈2.0` |
| `sim/tests/regress_gnd_cycle.py` | 低幅循环 GND/塑性累积趋势 | `sim/tests/regress_runs/2026-02-05/gnd_cycle/summary.json` | `passed=true`，`gnd_growth>0`，`accum_growth>0` |
| `sim/tests/regress_bc_crack.py` | 边界处理 + Mode-I 裂纹增量（小规模） | `sim/tests/regress_runs/2026-02-05/bc_crack/summary.json` | `passed=true`，压缩不误增裂 |
| `sim/tests/regress_bc_crack_large.py` | 大网格边界/裂纹门禁 | `sim/tests/regress_runs/2026-02-05/bc_crack_large/summary.json` | `passed=true`，`modeI_phi_mean_delta>0` |
| `sim/tests/regress_bc_crack_micron.py` | 微米尺度缩放一致性 | `sim/tests/regress_runs/2026-02-05/bc_crack_micron/summary.json` | `passed=true`，阈值全满足 |
| `sim/tests/regress_all.py` | 边界裂纹三件套统一入口 | `sim/tests/regress_runs/2026-02-06/phase1_baseline/boundary_crack/summary.json` | `passed=true`，`small/large/micron` 全通过 |
| `sim/tests/run_phase1_suite.py` | Phase-1 一键全回归 | `sim/tests/regress_runs/2026-02-06/phase1_suite_venvcheck/summary.json` | `passed=true`，fresh venv 可复现 |
| `sim/tests/regress_phase2.py` | Phase-2 门禁编排（onset + 对齐 + 可扩展 gate） | `sim/tests/regress_runs/2026-02-09/release_d2_d3_quick_seedfull/phase2_gate/summary.json` | `passed=true`，`with_exp_alignment=true`，`with_d2_localization=true` |
| `sim/tests/regress_exp_alignment.py` | 单工况实验对齐阈值门禁 | `sim/tests/regress_runs/2026-02-09/release_d2_d3_quick_seedfull/phase2_gate/exp_alignment/summary.json` | `passed=true`，`rmse_tau≈28.45 MPa`，`rmse_gamma≈0.003889` |
| `sim/tests/regress_energy_consistency.py` | 周期能量/裂纹/塑性单调性门禁 | `sim/tests/regress_runs/2026-02-08/energy_gate_smoke/summary.json` | `passed=true`，`energy_drop_count=0`，`reversal_count=0` |
| `sim/tests/regress_exp_alignment_multi.py` | 多工况实验对齐门禁 | `sim/tests/regress_runs/2026-02-08/exp_alignment_multi_week8_smoke/summary.json` | `passed=true`，`5/5` 条件通过 |
| `sim/tests/repeat_crack_onset_seeds.py` | seed 稳健性批次门禁 | `sim/tests/regress_runs/2026-02-09/d1_full_gate/seed_batch_1/summary.json` | `passed=true`，`seed_gate_pass_count=3/3`（batch_2 同样通过） |
| `sim/tests/regress_d2_localization_energy.py` | D2：裂纹局部化 + 能量密度场完整性 | `sim/tests/regress_runs/2026-02-09/d2_localization_energy/summary.json` | `passed=true`，`localization_index_peak=5.1072` |
| `sim/tests/regress_d3_multiphysics_matrix.py` | D3：正向/负向矩阵门禁 | `sim/tests/regress_runs/2026-02-09/d3_multiphysics_matrix_smoke/summary.json` | `passed=true`，`passed_count=3/3` |
| `sim/tests/run_d1_full_gate.py` | D1：phase2_full + multi_align_full + seed_full 编排 | `sim/tests/regress_runs/2026-02-09/d1_full_gate/summary.json` | `passed=true`，验收矩阵全绿 |
| `sim/tests/run_release_baseline_week4.py` | release bundle 编排（Phase-2 + D2 + D3 + seed） | `sim/tests/regress_runs/2026-02-09/release_d2_d3_quick_seedfull/bundle_summary.json` | `passed=true`，D2/D3/seed 批次全部通过 |
| `sim/tests/run_ci_smoke.py` | CI 最小链路（phase2/multi_align/seed_ci） | `sim/tests/regress_runs/2026-02-08/ci_smoke_local_verify_fix/summary.json` | `passed=true`，三子任务全通过 |
| `sim/tests/run_seed_robustness_20.py` | 20-seed 分批模板执行 | `sim/tests/regress_runs/2026-02-08/seed_robustness_20_smoke_pair/bundle_summary.json` | `passed=true`（smoke 模式） |
| `sim/tests/summarize_seed_robustness_ci.py` | seed 批次统计 + Wilson CI 聚合 | `sim/tests/regress_runs/2026-02-08/seed_ci_summary_smoke/summary.json` | `all_seed_gate_passed=true`，`seed_unique_count=6` |

## 尚需完善（下一阶段）
- 将 D2/D3 从 quick profile 扩展到 full profile 常态门禁（保留 `release_d2_d3_quick_seedfull` 作为快速基线）。
- 完成 20-seed full-case 批次闭环（包含 CI 汇总 JSON/MD + 固化接受阈值）。
- 对 D3 负对照集合增加“中等驱动/不同晶向”子集，进一步验证“不过触发”稳健性。
- 增加长周期（>10 cycles）能量密度与裂纹局部化联合回归，补齐“时间尺度外推”证据链。
- 将 release 产物摘要自动写入统一 markdown（当前已有 `bundle_summary.json`，建议补标准化周报模板落盘）。
