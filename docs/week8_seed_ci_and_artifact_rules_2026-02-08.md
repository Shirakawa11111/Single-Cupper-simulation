# Week-8 Seed CI + Artifact Rules (2026-02-08)

## 1) 20 Seeds 分批模板

默认模板（`41-60`，每批 `5` 个，共 `4` 批）：

```bash
python sim/tests/run_seed_robustness_20.py \
  --base-config sim/configs/crack_onset_scan.yaml \
  --case-mode full \
  --seed-start 41 \
  --seed-count 20 \
  --batch-size 5 \
  --out-root sim/tests/regress_runs/$(date +%F)/seed_robustness_20_full
```

自定义 seed 列表示例：

```bash
python sim/tests/run_seed_robustness_20.py \
  --base-config sim/configs/crack_onset_scan.yaml \
  --case-mode full \
  --seeds 61,62,63,64,65,66,67,68,69,70 \
  --batch-size 5 \
  --out-root sim/tests/regress_runs/$(date +%F)/seed_robustness_custom
```

## 2) 只做 CI 汇总（不重跑仿真）

```bash
python sim/tests/summarize_seed_robustness_ci.py \
  --batch-glob "sim/tests/regress_runs/$(date +%F)/release_pack_v2/seed_batch_*" \
  --out sim/tests/regress_runs/$(date +%F)/seed_ci_summary/summary.json \
  --markdown-out sim/tests/regress_runs/$(date +%F)/seed_ci_summary/summary.md \
  --aggregate-csv-out sim/tests/regress_runs/$(date +%F)/seed_ci_summary/aggregate.csv
```

关键输出：
- `summary.json`：总体通过率 + Wilson 置信区间 + 数值指标均值区间。
- `summary.md`：可直接贴入周报/handoff 的表格化摘要。
- `aggregate.csv`：按 seed 合并后的明细，用于后续绘图或二次统计。

## 3) 产物管理规则（提交前）

当前 `.gitignore` 已覆盖：
- Python 缓存：`__pycache__/`, `*.py[cod]`
- 回归与实验产物目录：`sim/tests/runs/`, `sim/tests/regress_runs/`
- 临时与局部产物：`sim/tests/tmp/`, `sim/tests/virtual_cycle*.csv`, `sim/tests/virtual_cycle*.png`, `sim/tests/*_stress_strain.csv`, `sim/tests/*.stdout`, `sim/tests/*.stderr`

建议提交前检查：

```bash
git status --short
```

目标：只提交源码/配置/文档，不提交缓存与运行产物。

