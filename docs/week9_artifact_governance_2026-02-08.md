# Week-9 Artifact Governance (2026-02-08)

## 目标
- 固化运行产物分层规则，减少无效产物沉积。
- 通过统一清理脚本，保证日常 `git status` 可读且可控。

## 目录分层规则
### 1) 运行层（原始产物）
- 根目录：`sim/tests/runs/YYYY-MM-DD/`
- 用途：单次仿真、扫描、探索性任务的原始输出。
- 命名：`<task>_<HHMMSS>`（例如 `monotonic_baseline_114230`）。

### 2) 回归门禁层（可审计）
- 根目录：`sim/tests/regress_runs/YYYY-MM-DD/`
- 用途：Phase-1/Phase-2/CI smoke/release baseline 的可审计汇总。
- 命名：`<bundle_or_gate>` 或 `<bundle_or_gate>_<HHMMSS>`。

### 3) 发布层（对外交付）
- 建议目录：`sim/tests/regress_runs/YYYY-MM-DD/release_pack_v2_rerun_<HHMMSS>/`
- 最小交付集：
  - `phase2_full/summary.json`
  - `seed_batch_1/summary.json`
  - `seed_batch_2/summary.json`
  - `exp_alignment_multi/summary.json`
  - `release_pack_v2_report*.md`

## 重复与无效产物判定
- `*_rerun_<HHMMSS>` 存在时，视同名基目录为被替代版本。
- `*_verify_fix` 存在时，视同名 `*_verify` 为被替代版本。
- 同一天同任务 `*_HHMMSS` 多份目录，默认仅保留最新时间戳目录（当前脚本默认只对 `sim/tests/runs` 执行）。
- Python/macOS 缓存与临时文件（`__pycache__`、`.DS_Store`、`*.pyc` 等）统一判为无效产物。

## 清理脚本
- 脚本：`sim/tests/cleanup_artifacts.sh`
- 默认：`dry-run`，仅打印候选删除路径。
- 执行删除：`bash sim/tests/cleanup_artifacts.sh --apply`
- 扩展到 `regress_runs` 的时间戳去重：`bash sim/tests/cleanup_artifacts.sh --apply --include-regress-timestamp-dups`
- 旧日期保留策略（可选）：`bash sim/tests/cleanup_artifacts.sh --apply --retention-days 7 --keep-date 2026-02-08`

## 本轮落地说明（2026-02-09）
- 已新增清理脚本并启用以下默认保护：
  - 不删除 Git 已跟踪路径。
  - 不删除在 `README/HANDOFF/docs` 中被明确引用的路径（除非显式 `--force-referenced`）。
  - 跳过虚拟环境目录（`.venv*` / `venv`），避免误删运行环境。
- 推荐日常流程：
  1. 先执行 `bash sim/tests/cleanup_artifacts.sh` 检查候选；
  2. 确认后执行 `bash sim/tests/cleanup_artifacts.sh --apply`；
  3. 最后执行 `git status --short` 复核仅保留源码/配置/文档改动。

## 本轮执行记录
- Dry-run：`bash sim/tests/cleanup_artifacts.sh`
  - 候选总数：`111`
  - 主要类别：Python 缓存、`.DS_Store`、`*_verify` 被 `*_verify_fix` 替代、`*_rerun_*` 基目录替代、同任务旧时间戳目录。
- Apply：`bash sim/tests/cleanup_artifacts.sh --apply`
  - 实际清理：`111` 项。
- 复核：再次执行 `bash sim/tests/cleanup_artifacts.sh`
  - 输出：`No cleanup candidates.`
