# Handoff Notes

Project path: /Users/bojingkai/Desktop/单晶铜拉伸模拟
Date: 2026-02-06
Current branch/commit: main @ 24be1d1 (Phase-2 stability gate and crack-onset threshold lock)

Status
- Working tree is dirty with many generated artifacts (pycache, VTKs, test runs). Do not commit these.
- Use git status -sb to inspect; large untracked trees live under sim/tests/runs and sim/tests/regress_runs.
- Phase-1 reproducibility scaffolding is now in repo: `.gitignore` hygiene rules, root `requirements*.txt`, config-driven runners, and docs under `docs/`.

What was added recently (summary)
- Multi-slip plasticity state variables (gamma_s, chi_s, tau_c) and flow rule; plastic_vec from slip-rate contributions.
- GND diagnostics via new sim/dislocation.py (Nye tensor, gnd_density).
- PFC can be disabled with SolverConfig.pfc_active; optional GND diagnostics via gnd_active.
- virtual_cycle now outputs gnd_mean/gnd_max in CSVs; VTK includes gnd_density when present.
- New regression/utility scripts under sim/tests: regress_gnd, regress_gnd_cycle, scan_hgnd_cycle, scan_gnd_orientations, plot helpers.

Week-2 in-progress changes (2026-02-06)
- Stability instrumentation:
  - `sim/mechanics.py`: add `last_solve_info` (CG failures / outer convergence snapshot).
  - `sim/solver.py`: add step-level finite checks + `last_step_diagnostics`.
  - `sim/dislocation.py`: harden `gnd_density` against NaN/Inf.
  - `sim/tests/virtual_cycle.py`: aggregate per-step diagnostics into `diagnostics_out`.
- Runner/gate enhancements:
  - `sim/tests/run_virtual_cycle_config.py`: records runtime warnings + diagnostics + pass/fail reasons, with threshold flags.
  - `sim/tests/run_phase1_suite.py`: adds RuntimeWarning counting/threshold gating.
  - new `sim/tests/scan_crack_onset.py`: config-driven crack-onset scan with `summary.json`/`summary.csv`.
  - new `sim/tests/regress_phase2.py`: Phase-2 gate (Phase-1 suite + config run + crack-onset scan).
- New configs/docs:
  - `sim/configs/crack_onset_scan.yaml`
  - `sim/configs/crack_onset_aggressive_only.yaml`
  - `sim/configs/crack_onset_scan_quick.yaml`
  - `docs/calibration_phase2.md`
- Solver strategy updates (second pass):
  - mechanical iterative solve now supports residual acceptance, solution magnitude guard, optional GMRES fallback, and rigid-translation removal.
  - crack CG now supports residual/incomplete acceptance to avoid update stalls.

Key scripts to rerun
- python sim/tests/regress_microstrain.py
- python sim/tests/regress_gnd.py
- python sim/tests/regress_gnd_cycle.py
- python sim/tests/scan_hgnd_cycle.py
- python sim/tests/scan_gnd_orientations.py
- python sim/tests/run_virtual_cycle_config.py --config sim/configs/fatigue_lowamp.yaml
- python sim/tests/run_phase1_suite.py --strict --out sim/tests/regress_runs/YYYY-MM-DD/phase1_suite

Phase-1 files added
- requirements: `requirements.txt`, `requirements-dev.txt`
- configs: `sim/configs/monotonic_baseline.yaml`, `sim/configs/fatigue_lowamp.yaml`, `sim/configs/notch_gnd.yaml`
- runners: `sim/tests/run_virtual_cycle_config.py`, `sim/tests/run_phase1_suite.py`
- docs: `docs/phase1_baseline_2026-02-06.md`, `docs/units_mapping.md`, `docs/parameter_register.md`, `PHASE1_CHECKLIST.md`

Recent outputs (not committed)
- sim/tests/regress_runs/2026-02-06/phase1_suite/
- sim/tests/regress_runs/2026-02-06/phase1_baseline/
- sim/tests/runs/2026-02-05/gnd_orient_hgnd_sens_193218/
- sim/tests/runs/2026-02-05/gnd_orient_scan_lowamp_notch_180900/
- sim/tests/runs/2026-02-05/gnd_orient_scan_lowamp_nothing_181500/
- sim/tests/regress_runs/2026-02-05/gnd_cycle_hgnd_scan_4pt/
- sim/tests/regress_runs/2026-02-05/gnd_cycle_hgnd0/
- sim/tests/regress_runs/2026-02-05/gnd_cycle_hgnd1e-4/
- sim/tests/regress_runs/2026-02-06/monotonic_stability_check_summary.json
- sim/tests/regress_runs/2026-02-06/crack_onset_scan_quick_smoke/
- sim/tests/regress_runs/2026-02-06/phase2_gate_quickcheck/
- sim/tests/regress_runs/2026-02-06/phase2_gate_smoke_v2/
- sim/tests/regress_runs/2026-02-06/crack_onset_scan_full_locked/
- sim/tests/regress_runs/2026-02-06/crack_onset_aggressive_only_post_solver_v2/
- sim/tests/regress_runs/2026-02-06/phase2_gate_200918/
- sim/tests/regress_runs/2026-02-06/crack_onset_scan_full_locked_v2/
- sim/tests/regress_runs/2026-02-06/crack_onset_scan_full_locked_v3/

Phase-1 validation snapshot (2026-02-06)
- Strict suite command:
  - python sim/tests/run_phase1_suite.py --strict --out sim/tests/regress_runs/2026-02-06/phase1_suite
- Aggregate summary:
  - sim/tests/regress_runs/2026-02-06/phase1_suite/summary.json
- Result:
  - passed=true
  - total_s=92.458
  - microstrain=0.381s, gnd=0.117s, gnd_cycle=53.984s, boundary_crack=37.974s
- Baseline mirror:
  - sim/tests/regress_runs/2026-02-06/phase1_baseline/
- Config runner dry-runs:
  - monotonic_baseline.yaml / fatigue_lowamp.yaml / notch_gnd.yaml
- Config runner actual run:
  - monotonic_baseline.yaml (cycles_completed=1, duration_s=14.654)
  - summary: sim/tests/runs/2026-02-06/phase1_config_runs/monotonic_baseline_run_summary.json
  - fatigue_lowamp.yaml (accelerated baseline: cycles_completed=5, duration_s=597.483, stable stop at window=5)
  - summary: sim/tests/runs/2026-02-06/phase1_config_runs/fatigue_lowamp_run_summary.json
  - notch_gnd.yaml (cycles_completed=8, duration_s=508.516)
  - summary: sim/tests/runs/2026-02-06/phase1_config_runs/notch_gnd_run_summary.json

Numerical notes
- Both fatigue_lowamp/notch_gnd runs emitted RuntimeWarning from scipy CG and gradient ops.
- Runs completed and summaries contain finite last-cycle metrics; keep warnings as a follow-up numerical-stability task.
- `scan_crack_onset.yaml` 单 case（control_notch_mild）出现大量 RuntimeWarning（主要为 invalid value in add/subtract）且机械 CG 非零信息频繁：
  - 见 `sim/tests/regress_runs/2026-02-06/phase2_gate_smoke_v2/crack_onset_scan/summary.json`
  - 该结果用于定位 Phase-2 数值稳定问题，不是基线通过记录。
- 快速烟测链路可通过（用于开发验证，不代表物理收敛）：
  - `sim/tests/regress_runs/2026-02-06/phase2_gate_quickcheck/summary.json`
- Final threshold-lock scan (`crack_onset_scan_full_locked`) passes with:
  - `onset_cases=3/4`
  - `runtime_warning_count=0` for all cases
  - criteria currently locked at `min_onset_cases=1`, `max_runtime_warnings=50`
- Criteria-refresh full scan (`crack_onset_scan_full_locked_v3`) passes with:
  - `onset_cases=3/4`
  - `runtime_warning_count=0` for all cases
  - length-led onset logic enabled (`onset_length` primary, `onset_mean_aux` auxiliary)
  - criteria currently locked at:
    - `min_crack_delta=5.0e-2`
    - `min_crack_mean_delta=5.0e-4`
    - `min_crack_length_for_mean_aux=1.0e-1`
    - `max_runtime_warnings=50`
    - `max_mechanical_cg_failures=1300`
    - `max_mechanical_not_accepted_steps=320`
    - `max_crack_cg_nonconverged_steps=320`
    - `max_nonfinite_count=0`
- Residual risk:
  - notch cases still show frequent mechanical `info>0` and high `mechanical_not_accepted_steps`.
  - keep this as next numerical-robustness target (preconditioned operator or alternative linear solver for unilateral branch).

Open decisions / next steps
- [Resolved for Phase-1] Physical reference mapping:
  - L0=1e-6 m, sigma_ref=168.4 GPa, b_phys=2.556e-10 m
  - recommended gnd_burgers_nd=2.556e-4 for SI GND reporting
  - see docs/units_mapping.md and docs/parameter_register.md
- Decide whether to seed initial GND before fatigue (gamma_s gradient or beta_p init).
- Decide whether to output gnd_density_abs (L1) in addition to gnd_density (L2 norm).
- Optional: add COMSOL bridge usage once a model exists.
- Prioritize mechanical CG diagnostics:
  - confirm whether high `mechanical_cg_failures` reflects true non-convergence vs expected null-space behavior.
  - if needed, add preconditioner/regularization or solver fallback for unilateral branch.
- Continue tightening Phase-2 thresholds after solver improvements:
  - lower `max_mechanical_not_accepted_steps`
  - lower `max_crack_cg_nonconverged_steps`

Notes
- accum_plastic is currently sum|gamma_s|; eps_eq can stay as output-only.
- GND uses Nye tensor curl of beta_p; gnd_density is magnitude (sign-canceling only happens if you average signed components).
