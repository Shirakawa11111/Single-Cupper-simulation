# Week 8 Multi-Condition Alignment Summary (2026-02-08)

Source summary:
- `sim/tests/regress_runs/2026-02-08/exp_alignment_multi_week8_smoke/summary.json`

## Run Setup
- Config: `sim/configs/exp_alignment_multi_week8.yaml`
- Conditions: 5 (`cycle=208,1000,1894,2379,3701`)
- Reuse strategy: `reuse_first_sim_csv=true` (首工况完整仿真，后续工况复用 `sim_csv`)
- Result: `passed_count=5/5`, `passed=true`

## Condition Metrics

| Condition | RMSE τ (MPa) | MAE τ (MPa) | RMSE γ | MAE γ | Passed |
|---|---:|---:|---:|---:|---|
| lowamp_cycle_0208 | 26.7735 | 21.7043 | 0.003706 | 0.002997 | true |
| lowamp_cycle_1000 | 28.4474 | 23.1806 | 0.003889 | 0.003141 | true |
| lowamp_cycle_1894 | 28.4867 | 23.3453 | 0.003880 | 0.003146 | true |
| lowamp_cycle_2379 | 28.2288 | 23.1790 | 0.003850 | 0.003130 | true |
| lowamp_cycle_3701 | 26.9441 | 22.2079 | 0.003714 | 0.003041 | true |

## Aggregated Metrics
- Avg RMSE τ: `27.7761 MPa`
- Avg MAE τ: `22.7234 MPa`
- Avg RMSE γ: `0.00380785`
- Avg MAE γ: `0.00309109`

## Notes
- 当前 5 工况均来自同一实验批次（`-111` 取向），覆盖不同循环数。
- 这一步验证了“多工况接口 + 复用 sim_csv 加速”可稳定工作。
