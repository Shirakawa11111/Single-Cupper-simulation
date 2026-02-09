# D1 Full-Gate Progress (2026-02-09)

## 已完成实现
- 新增真实多工况配置：`sim/configs/exp_alignment_multi_d1_full.yaml`（5 条件，`min_pass_count=5`）。
- 新增 D1 编排脚本：`sim/tests/run_d1_full_gate.py`（统一 `acceptance` + `failure_reasons`）。
- 新增 D1 一键命令：`sim/tests/d1_full_gate_commands.sh`。
- 锁参接入：`sim/configs/release_pack_v2_lock.yaml` 增加 `multi_alignment_full`。

## 本地验证记录
1) D1 编排 smoke（快速配置，严格阈值）  
- 输出：`sim/tests/regress_runs/2026-02-09/d1_full_gate_smoke_verify/summary.json`  
- 结论：失败（`phase2_task_failed:exp_alignment`），失败根因已按预期写入 `failure_reasons`。

2) D1 编排 smoke（快速配置，放宽对齐阈值用于流程验证）  
- 输出：`sim/tests/regress_runs/2026-02-09/d1_full_gate_smoke_verify_pass/summary.json`  
- 结论：通过（`passed=true`），`acceptance.phase2_full/multi_align_full` 均为 `true`。

3) D1 真实多工况配置单独验证（5 条件）  
- 输出：`sim/tests/regress_runs/2026-02-09/exp_alignment_multi_d1_full_verify/summary.json`  
- 结论：`passed_count=5/5`，`passed=true`。

## 下一步（D1-5）
- 执行首轮完整 D1（非 quick、全 case、含 seed 批次）：
```bash
bash sim/tests/d1_full_gate_commands.sh
```
- 目标汇总路径：`sim/tests/regress_runs/<date>/d1_full_gate/summary.json`

## D1-5 实际执行结果（2026-02-09）
- 汇总：`sim/tests/regress_runs/2026-02-09/d1_full_gate/summary.json`
- 总结论：`passed=true`，`failure_reasons=[]`。
- 验收矩阵：
  - `acceptance.phase2_full.passed=true`
  - `acceptance.multi_align_full.passed=true`（`condition_total=5`，`passed_count=5`）
  - `acceptance.seed_robustness.passed=true`（`seed_gate_pass_count=6/6`）
