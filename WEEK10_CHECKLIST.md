# Week 10 Checklist (D1 Full-Gate)

说明：
- `⭕` 已完成
- `○` 待完成

## 1. 清单
- ⭕ D1-1 建立真实多工况配置（5 条件，非 skeleton）。
- ⭕ D1-2 建立 D1 full-gate 编排脚本（统一 summary + failure_reasons）。
- ⭕ D1-3 建立 D1 一键命令包（默认 full-case seed 模式）。
- ⭕ D1-4 将 D1 配置写入锁参文件（兼容保留 skeleton）。
- ⭕ D1-5 运行首轮 D1 full-gate（非 quick、全 case、含 seed 批次）。

结果路径：
- `sim/tests/regress_runs/2026-02-09/d1_full_gate/summary.json`（`passed=true`）

## 2. 验收标准
- `phase2_full/summary.json` 中 `passed=true`，且包含 `with_exp_alignment=true` 与 `with_energy_gate=true`。
- `exp_alignment_multi/summary.json` 中 `condition_total>=3` 且 `passed_count==condition_total`。
- `seed_batch_*/summary.json` 中 `all_seed_gate_passed=true`。
- 顶层 `summary.json` 输出统一验收矩阵：`acceptance`，并输出失败根因：`failure_reasons`。

## 3. 目标产物路径
- `sim/configs/exp_alignment_multi_d1_full.yaml`
- `sim/tests/run_d1_full_gate.py`
- `sim/tests/d1_full_gate_commands.sh`
- `sim/tests/regress_runs/<date>/d1_full_gate/summary.json`
- `sim/tests/regress_runs/<date>/d1_full_gate/phase2_full/summary.json`
- `sim/tests/regress_runs/<date>/d1_full_gate/exp_alignment_multi/summary.json`
- `sim/tests/regress_runs/<date>/d1_full_gate/seed_batch_1/summary.json`
- `sim/tests/regress_runs/<date>/d1_full_gate/seed_batch_2/summary.json`
