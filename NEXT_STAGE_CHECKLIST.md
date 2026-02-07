# Next Stage Checklist (Week 5-7)

说明：
- `⭕` 已完成
- `○` 待完成

## A. 稳健性与质量门禁（Week 5）
- ⭕ 1) 建立下一阶段总清单 + 明确每项验收标准（本文件）。
- ⭕ 2) 将 `repeat_crack_onset_seeds.py` 扩展为“全 case 复验模式”（3 个 notch + 负对照）。
- ⭕ 3) 运行两批次全 case seed 复验并形成统计汇总（目标至少 6 seeds）。
- 批次结果：`batch1(41,42,43)=3/3`，`batch2(44,45,46)=3/3`，总计 `6/6` 通过。
- ⭕ 4) 将全 case seed 复验接入 `run_release_baseline_week4.py`（可选开关）。
- 集成 smoke：`release_baseline_week4_quick_seedfull_smoke` 通过（`seed_case_mode=full`，`passed=true`）。

## B. 物理一致性增强（Week 6）
- ⭕ 5) 新增能量判据回归脚本（裂纹驱动力/能量项趋势、非物理反转检测）。
- ⭕ 6) 将能量判据作为可选 gate 接入 `regress_phase2.py` 并验证。
- 集成 smoke：`phase2_with_energy_gate_smoke` 通过（`with_energy_gate=true`，总门禁通过）。

## C. 研究交付准备（Week 7）
- ⭕ 7) 建立“多工况对齐接口”与配置骨架（当前数据仍以单工况为主，先打通流程）。
- 新增：`sim/tests/regress_exp_alignment_multi.py` + `sim/configs/exp_alignment_multi_skeleton.yaml`，单工况 smoke 已通过。
- ⭕ 8) 输出 `release pack v2`（配置锁参 + 全门禁命令 + 自动报告模板 + handoff）。
- 产物：`sim/configs/release_pack_v2_lock.yaml`、`sim/tests/release_pack_v2_commands.sh`、`sim/tests/release_pack_v2_report_template.md`、`sim/tests/build_release_pack_v2_report.py`、`HANDOFF_RELEASE_PACK_V2.md`。
