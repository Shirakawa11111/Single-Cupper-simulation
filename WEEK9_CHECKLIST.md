# Week 9 Checklist

说明：
- `⭕` 已完成
- `○` 待完成

## 1. 清单
- ⭕ A1 最小 CI 门禁落地（`phase2_quick + multi_align_smoke + seed_ci_smoke`）。
- ⭕ A2 CI 失败信息统一聚合为单一汇总 JSON。
- ⭕ B1 依赖锁定文件落地（在现有 `requirements*.txt` 基础上增加锁定版本）。
- ⭕ B2 新环境一键复现文档落地（创建 venv、安装、最小回归命令）。
- ⭕ C1 完成 nd→SI 样例（补齐 `docs/units_mapping.md` 未完成项）。
- ⭕ C2 将 nd→SI 样例接入对齐报告模板（统一汇报口径）。
- ⭕ D1 运行产物分层治理规则落地（`runs/`、`regress_runs/`、发布包）。
- ⭕ D2 产物清理脚本落地并清理无效/重复目录。

## 2. 验收标准
- A 模块：PR 触发后可自动得到 `passed/failed + failure_reasons`，并附关键日志索引。
- B 模块：在新建 venv 场景下可一键安装并跑通最小门禁，结果与基线一致（允许浮点微差）。
- C 模块：样例必须包含输入字段、换算公式、输出字段、单位与可复算过程。
- D 模块：日常运行后 `git status` 不出现大规模无关产物变动，目录层级清晰可追溯。

## 3. 目标产物路径
- `sim/tests/regress_runs/<date>/ci_smoke/summary.json`
- `requirements-lock.txt`
- `docs/week9_reproducibility_YYYY-MM-DD.md`
- `docs/units_mapping.md`
- `docs/week9_units_example_YYYY-MM-DD.md`
- `docs/week9_artifact_governance_YYYY-MM-DD.md`
- `sim/tests/cleanup_artifacts.sh`
