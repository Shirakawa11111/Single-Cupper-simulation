# Week-9 Units Conversion Worked Example (2026-02-08)

## 目标
- 用真实回归产物字段给出 `nd -> SI` 的可复算样例。
- 给对齐报告提供统一的 `MPa <-> nd` 换算口径。

## 统一参考常数
- `L0 = 1.0e-6 m`
- `sigma_ref = 168.4 GPa = 168400 MPa`
- `b_phys = 2.556e-10 m`
- `gnd_burgers_nd = 2.556e-4 (= b_phys / L0)`

## 样例 A：GND 行（`nd -> SI`）
- 输入来源：`sim/tests/regress_runs/2026-02-06/phase1_suite_venvcheck/gnd_cycle/summary.json`
- 输入字段：`results.gnd_means[4] = 2.6317655902047965e-05`（无量纲代码值）

换算公式：
- 通用：`rho_phys = rho_code * gnd_burgers_nd / (L0 * b_phys)`
- 推荐简化：`rho_phys = rho_code / L0^2`（当 `gnd_burgers_nd = b_phys / L0`）

输出：
- `rho_phys = 2.6317655902047965e-05 / (1.0e-6)^2 = 2.6317655902047965e+07 m^-2`

## 样例 B：对齐应力误差（`MPa <-> nd`）
- 输入来源：`sim/tests/regress_runs/2026-02-08/exp_alignment_multi_week8_smoke/conditions/lowamp_cycle_1000.summary.json`
- 输入字段：`metrics.rmse_tau_MPa = 28.44743777411884 MPa`

换算公式：
- `tau_MPa = tau_nd * sigma_ref_MPa`
- `tau_nd = tau_MPa / sigma_ref_MPa`

输出：
- `rmse_tau_nd = 28.44743777411884 / 168400 = 1.689277777560501e-04`

## 一键复算（Python）
```bash
python - <<'PY'
rho_code = 2.6317655902047965e-05
L0 = 1.0e-6
rmse_tau_mpa = 28.44743777411884
sigma_ref_mpa = 168400.0

rho_phys = rho_code / (L0 ** 2)
rmse_tau_nd = rmse_tau_mpa / sigma_ref_mpa

print("rho_phys_m^-2 =", rho_phys)
print("rmse_tau_nd =", rmse_tau_nd)
PY
```

## 报告接入约定
- 对齐报告保留 SI 指标（`MPa` / `gamma`）用于业务沟通。
- 同时附一列 `nd` 指标（至少 `rmse_tau_nd`），用于与求解器无量纲参数直观对齐。
