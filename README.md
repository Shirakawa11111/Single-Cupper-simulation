# 单晶铜拉伸/疲劳相场项目说明

## 概览
- 目标：在 3D 下耦合 PFC 密度、损伤/裂纹、等效塑性与力学平衡，模拟单晶铜（及未来多晶）在循环载荷中的裂纹萌生与疲劳规律，并输出可视化/统计用于 Paris、Coffin–Manson 对比。
- 主要变量：PFC 密度 ψ，裂纹场 φ，等效塑性标量与方向分量，位移场（微观校正 + 宏观总位移），应力张量。
- 能量框架：弹性正/负能分裂 + 断裂能（韧性随等效塑性退化） + PFC 能 + 取向/晶界项；自由能变分得到化学势，PFC 动力学 ∂tψ = -∇²μ。

## 代码结构（关键文件）
- `sim/energy.py`：能量项、韧性退化、塑性/方向性驱动；`plastic_measures` 计算等效塑性与方向分量（机械 von Mises + PFC 梯度混合）。
- `sim/solver.py`：交替求解器（力学 → 塑性松弛 → 裂纹 → PFC），支持方向性驱动、应力耦合 μ_extra，跟踪 stress/stress_vm、plastic_vec。
- `sim/io.py`：LAMMPS/VTK 输出（VTK 现为二进制 STRUCTURED_GRID，可选变形坐标），附带归一化场与应力、塑性方向分量。
- `sim/tests/virtual_cycle.py`：虚拟循环载荷驱动脚本（对称三角波），记录 CSV、VTK、LAMMPS，拟合 Paris/Coffin–Manson 斜率。
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
- VTK：`sim/tests/virtual_cycle_vtk/anim_frame_*.vtk`（二进制，STRUCTURED_GRID，含 crack/plastic/psi/stress/displacement 等及归一化字段）。
- LAMMPS dump：`sim/tests/virtual_cycle_*.lammpstrj`（包含 crack, plastic, psi, plastic_vec 分量, stress_vm, stress_xx/yy/zz）。
- CSV：`sim/tests/virtual_cycle.csv`（cycle, energy, crack_mean, plastic_mean, crack_delta, plastic_range）。

可视化提示：
- ParaView/Ovito 固定色标，禁用 per-timestep rescale。裂纹用 `crack_clamp03` (0–0.3) 或 `crack_norm` (0–1)；塑性/应力用 0–1；位移查看 `displacement_total`/`disp_total_norm` 或 warp by total displacement。

## 最近更新（对话内已完成）
- 塑性/方向场：新增 `plastic_measures`，机械 von Mises/轴向应变与 PFC 梯度按权重混合（默认机械占比 0.9），输出塑性向量。
- 方向性裂纹驱动：加载轴塑性分量放大历史能量/驱动力（`dir_coupling` 默认 0.8）。
- 应力耦合：von Mises 应力归一化后加入 μ_extra，PFC 在高应力区更敏感。
- 循环载荷：对称三角波 0→+ε_max→0→−ε_max→0，每段 50 步，支持失败阈值提前停，记录每周期 crack/plastic 的均值与增量/范围。
- 输出：VTK 改为二进制 STRUCTURED_GRID，附归一化场；LAMMPS 去重了多余的 plastic_vec 列。
- 报告：`report.tex` 同步上述公式/流程/输出说明，添加当前结果快照。

## 当前结果快照（默认参数：128×64×16，缺陷幅值 0.12，ε_max=0.08，dir_coupling=0.8）
- 裂纹：最高 ~0.034，轻微均匀损伤，未贯穿。
- 塑性：标量 ~0.7–0.9 累积，方向分量卸载后接近 0。
- 应力/位移：加载时宏观梯度明显；卸载到 0 时应力/位移回零，数值稳定。
- 可视化：二进制 VTK 无读错，固定色标可避免“全红”假象。

## 待拓展
- 增载或延长循环/调韧性与缺陷以触发裂纹局部化；输出能量密度场（弹性能/断裂能）用于裂尖能量分析。
- 引入多晶/晶界取向场与非周期边界（FFT/FEM 替换 CG）。
