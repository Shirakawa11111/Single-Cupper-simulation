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

## Week 9 (candidate)
### 1. 清单
- [x] ~~A1 最小 CI 门禁落地（`phase2_quick + multi_align_smoke + seed_ci_smoke`）。~~
- [x] ~~A2 CI 失败信息统一聚合为单一汇总 JSON。~~
- [x] ~~B1 依赖锁定文件落地（在现有 `requirements*.txt` 基础上增加锁定版本）。~~
- [x] ~~B2 新环境一键复现文档落地（创建 venv、安装、最小回归命令）。~~
- [x] ~~C1 完成 nd→SI 样例（补齐 `docs/units_mapping.md` 未完成项）。~~
- [x] ~~C2 将 nd→SI 样例接入对齐报告模板（统一汇报口径）。~~
- [x] ~~D1 运行产物分层治理规则落地（`runs/`、`regress_runs/`、发布包）。~~
- [x] ~~D2 产物清理脚本落地并清理无效/重复目录。~~

### 2. 验收标准
- A 模块：PR 触发后可自动得到 `passed/failed + failure_reasons`，并附关键日志索引。
- B 模块：在新建 venv 场景下可一键安装并跑通最小门禁，结果与基线一致（允许浮点微差）。
- C 模块：样例必须包含输入字段、换算公式、输出字段、单位与可复算过程。
- D 模块：日常运行后 `git status` 不出现大规模无关产物变动，目录层级清晰可追溯。

### 3. 目标产物路径
- `sim/tests/regress_runs/<date>/ci_smoke/summary.json`
- `requirements-lock.txt`
- `docs/week9_reproducibility_YYYY-MM-DD.md`
- `docs/units_mapping.md`
- `docs/week9_units_example_YYYY-MM-DD.md`
- `docs/week9_artifact_governance_YYYY-MM-DD.md`
- `sim/tests/cleanup_artifacts.sh`

## Week 10 (D1 full-gate candidate)
### 1. 清单
- [x] ~~D1-1 建立真实多工况配置（5 条件，非 skeleton）：`sim/configs/exp_alignment_multi_d1_full.yaml`。~~
- [x] ~~D1-2 建立 D1 full-gate 编排脚本（统一 summary + failure_reasons）：`sim/tests/run_d1_full_gate.py`。~~
- [x] ~~D1-3 建立 D1 一键命令包：`sim/tests/d1_full_gate_commands.sh`。~~
- [x] ~~D1-4 将 D1 多工况配置接入锁参：`sim/configs/release_pack_v2_lock.yaml`。~~
- [x] ~~D1-5 跑通首轮 full-gate（非 quick、全 case、含 seed full-case）并归档结果。~~

### 2. 验收标准
- `phase2_full`：`passed=true` 且 `with_exp_alignment=true`、`with_energy_gate=true`。
- `multi_align_full`：`condition_total>=3` 且 `passed_count==condition_total`。
- `seed_robustness`：两批次均 `all_seed_gate_passed=true`。
- 顶层 `summary.json` 必须输出 `acceptance` 与 `failure_reasons`。

### 3. 目标产物路径
- `WEEK10_CHECKLIST.md`
- `docs/d1_full_gate_checklist_2026-02-09.md`
- `sim/configs/exp_alignment_multi_d1_full.yaml`
- `sim/tests/run_d1_full_gate.py`
- `sim/tests/d1_full_gate_commands.sh`
- `sim/tests/regress_runs/<date>/d1_full_gate/summary.json`
