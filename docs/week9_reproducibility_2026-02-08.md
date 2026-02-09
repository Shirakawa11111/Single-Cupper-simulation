# Week-9 Reproducibility Guide (2026-02-08)

## 目标
- 固化一套可复现的最小运行环境与最小门禁链路。
- 输出单一汇总 JSON，便于本地与 CI 对比。

## 锁定依赖
- 运行时锁定文件：`requirements-lock.txt`
- 当前锁定版本：
  - `numpy==1.26.4`
  - `scipy==1.13.1`
  - `matplotlib==3.9.2`
  - `PyYAML==6.0.1`

## 新环境复现步骤
```bash
python -m venv .venv_week9
source .venv_week9/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

## 最小门禁命令（本地）
```bash
python sim/tests/run_ci_smoke.py \
  --out-root sim/tests/regress_runs/2026-02-08/ci_smoke_local
```

## 预期产物
- 汇总：`sim/tests/regress_runs/2026-02-08/ci_smoke_local/summary.json`
- 任务日志：`sim/tests/regress_runs/2026-02-08/ci_smoke_local/logs/*.stdout|*.stderr`
- 子任务摘要：
  - `phase2_quick`: `.../phase2_quick/summary.json`
  - `multi_align_smoke`: `.../multi_align_smoke/summary.json`
  - `seed_ci_smoke`: `.../seed_ci_smoke/summary.json`

## 说明
- `multi_align_smoke` 使用仓库内 fixture（`sim/tests/fixtures/exp_alignment_ci/`），不依赖本地私有实验目录。
- `seed_ci_smoke` 使用仓库内 fixture（`sim/tests/fixtures/seed_ci_smoke/`），用于验证 CI 统计链路。
- CI 工作流入口：`.github/workflows/ci-smoke.yml`。
