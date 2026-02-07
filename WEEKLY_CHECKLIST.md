# Weekly Checklist

## Week 1
- [x] ~~仓库卫生与可复现实验环境（`.gitignore` / `requirements*.txt`）~~
- [x] ~~配置化运行入口（`sim/configs/*.yaml` + `run_virtual_cycle_config.py`）~~
- [x] ~~一键 Phase-1 套件（`run_phase1_suite.py --strict`）~~
- [x] ~~README/HANDOFF/参数台账首轮同步~~

## Week 2
- [x] ~~Phase-2 门禁搭建（`regress_phase2.py` + `scan_crack_onset.py`）~~
- [x] ~~全扫描阈值收敛（`max_runtime_warnings` / `min_onset_cases`）~~
- [x] ~~机械分支稳定化（unilateral + jacobi + clip 策略）~~
- [x] ~~`max_mechanical_not_accepted_steps` 稳定压到 `<160` 且不回退~~

## Week 3
- [x] ~~降裁剪率专项：Notch 总裁剪步压到 `<80/160`（当前 `78/960`）~~
- [x] ~~收紧 crack-CG：从 80 -> 40 -> 20，并保持通过~~
- [x] ~~判据回正：Notch 保持 `onset_length=true` 主导~~
- [x] ~~实验对齐回归 + 对比图表（`regress_exp_alignment.py`）~~
- [x] ~~整链路门禁合并（`regress_phase2.py --with-exp-alignment`）并全量通过~~

## Next (Week 4 candidate)
- [x] ~~在保持当前数值门禁通过的前提下，做物理拟合专项：将 `rmse_tau_MPa` 从 `29.39` 降到 `28.45`，并保持 `rmse_gamma` 不劣化（`3.909e-3 -> 3.889e-3`）。~~
- [x] ~~增加“负对照不萌生 + 缺口样本萌生”的统计稳健性复验（两批次共 6 seeds，`seed_gate_pass=6/6`）。~~
- [x] ~~把当前锁参固化为“发布基线包”（配置 + 门禁命令 + 对齐报告模板）。~~
