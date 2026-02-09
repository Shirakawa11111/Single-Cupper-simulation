# Week 8 Checklist

说明：
- `⭕` 已完成
- `○` 待完成

## A. 多工况物理验证
- ⭕ 1) 将实验对齐从单工况扩展到 5 个循环工况并完成门禁跑通。
- ⭕ 2) 形成多工况误差汇总表（RMSE/MAE）并写入报告。
- 结果：`sim/tests/regress_runs/2026-02-08/exp_alignment_multi_week8_smoke/summary.json`，`passed_count=5/5`。
- 报告：`docs/week8_multi_condition_summary_2026-02-08.md`。

## B. 标定自动化准备
- ⭕ 3) 产出参数扫描/拟合脚本骨架（输入：配置网格，输出：最优参数与排名）。
- ⭕ 4) 将标定结果自动回写为锁参配置草案（不覆盖主锁参）。
- 脚本：`sim/tests/sweep_calibration_multi.py`（已完成 `max-runs=1` 烟测）。
- 输出：`sim/tests/regress_runs/2026-02-08/calibration_multi_week8_smoke/runs.csv`、`sim/configs/fatigue_lowamp_align_lock_draft_week8.yaml`。

## C. 统计稳健性
- ⭕ 5) 扩展 seed 复验批次模板到 20 seeds（分批执行），输出置信区间统计脚本。
- 模板：`sim/tests/run_seed_robustness_20.py`（默认 41-60，按 5 seeds/批执行）。
- 统计：`sim/tests/summarize_seed_robustness_ci.py`（Wilson CI + 汇总 Markdown）。
- 烟测：`sim/tests/regress_runs/2026-02-08/seed_robustness_20_smoke_pair/bundle_summary.json`（端到端通过）。
- 聚合验证：`sim/tests/regress_runs/2026-02-08/seed_ci_summary_smoke/summary.json`（基于两批次 6 seeds 聚合通过）。

## D. 工程化
- ⭕ 6) 增补 `.gitignore` 与运行产物管理规则，防止缓存/产物污染提交。
- `.gitignore` 新增：`sim/tests/*.stdout`、`sim/tests/*.stderr`、本地图片排除项。
- 已从版本库移除历史缓存/产物跟踪：`*.pyc`、`.DS_Store`、`sim/tests/virtual_cycle_stress_strain.*`。
- 规则文档：`docs/week8_seed_ci_and_artifact_rules_2026-02-08.md`。
