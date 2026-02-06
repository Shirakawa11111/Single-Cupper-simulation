# Handoff Notes

Project path: /Users/bojingkai/Desktop/单晶铜拉伸模拟
Date: 2026-02-06
Current branch/commit: main @ 724231c (Add h_gnd scan and orientation sensitivity outputs)

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

Open decisions / next steps
- [Resolved for Phase-1] Physical reference mapping:
  - L0=1e-6 m, sigma_ref=168.4 GPa, b_phys=2.556e-10 m
  - recommended gnd_burgers_nd=2.556e-4 for SI GND reporting
  - see docs/units_mapping.md and docs/parameter_register.md
- Decide whether to seed initial GND before fatigue (gamma_s gradient or beta_p init).
- Decide whether to output gnd_density_abs (L1) in addition to gnd_density (L2 norm).
- Optional: add COMSOL bridge usage once a model exists.

Notes
- accum_plastic is currently sum|gamma_s|; eps_eq can stay as output-only.
- GND uses Nye tensor curl of beta_p; gnd_density is magnitude (sign-canceling only happens if you average signed components).
