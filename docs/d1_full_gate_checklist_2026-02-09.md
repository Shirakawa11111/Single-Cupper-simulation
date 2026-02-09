# D1 Full-Gate Checklist (2026-02-09)

说明：
- `⭕` 已完成
- `○` 待完成

## 1. 清单
- ⭕ 建立 D1 多工况真实配置（5 条件）：`sim/configs/exp_alignment_multi_d1_full.yaml`。
- ⭕ 建立 D1 full-gate 编排脚本：`sim/tests/run_d1_full_gate.py`。
- ⭕ 建立 D1 一键命令包：`sim/tests/d1_full_gate_commands.sh`。
- ⭕ 将 D1 多工况配置接入锁参：`sim/configs/release_pack_v2_lock.yaml`。
- ⭕ 执行首轮 D1 full-gate（非 quick、全 case、含 seed full-case）。

执行结果：
- `sim/tests/regress_runs/2026-02-09/d1_full_gate/summary.json`（`passed=true`）。

## 2. 验收标准
- `phase2_full`：`passed=true`，并包含 `with_exp_alignment=true` 与 `with_energy_gate=true`。
- `multi_align_full`：`condition_total>=3` 且 `passed_count==condition_total`。
- `seed_robustness`：所有批次 `all_seed_gate_passed=true`。
- 汇总产物包含统一通过矩阵与失败根因：`acceptance` + `failure_reasons`。

## 3. 目标产物路径
- 编排入口：`sim/tests/run_d1_full_gate.py`
- 一键命令：`sim/tests/d1_full_gate_commands.sh`
- 多工况配置：`sim/configs/exp_alignment_multi_d1_full.yaml`
- D1 汇总：`sim/tests/regress_runs/<date>/d1_full_gate/summary.json`
- D1 任务日志：`sim/tests/regress_runs/<date>/d1_full_gate/logs/*.stdout|*.stderr`
- D1 子任务摘要：
  - `.../phase2_full/summary.json`
  - `.../exp_alignment_multi/summary.json`
  - `.../seed_batch_1/summary.json`
  - `.../seed_batch_2/summary.json`
