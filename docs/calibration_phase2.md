# Phase-2 Calibration Loop (SI/Experiment Alignment)

## Objective
- Freeze a reproducible loop from experiment -> nondimensional fit -> SI report:
  - cyclic hysteresis shape (`sigma_xx` vs `epsilon`)
  - crack-onset indicator (`crack_mean` / `crack_length`)
  - slip/GND trend (`accum_plastic_mean`, `gnd_mean`)

## Inputs (Required)
- Experimental CSV with columns:
  - `cycle`
  - `eps_axial` (strain, SI)
  - `sig_axial_MPa` (stress, SI)
- Crystal orientation and loading axis.
- Baseline units from `docs/units_mapping.md`.

## Parameter Groups
- Elastic:
  - `CopperParameters.c11/c12/c44`
- Plastic/slip:
  - `yield_tau`, `gamma0`, `slip_exponent`, `h_iso`, `h_gnd`
- Fracture:
  - `toughness_scale`, `gc`, `l0`, `epsilon_half`, `gres`
- Solver numerics:
  - `mech_max_iters`, `mech_tol`, `mech_outer_max_iters`, `mech_outer_tol`

## Execution Steps
1. Run baseline gate (must pass before fitting):
```bash
python sim/tests/regress_phase2.py --strict
```
2. Select candidate load path and run config:
```bash
python sim/tests/run_virtual_cycle_config.py \
  --config sim/configs/fatigue_lowamp.yaml \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 80 \
  --max-nonfinite-count 0
```
3. Compare simulation curve (`virtual_cycle_stress_strain.csv`) with experiment:
- Match small-strain slope first.
- Then match loop width (`plastic_range`) and RSS peak trend.
4. Run crack-onset screening after each parameter update:
```bash
python sim/tests/scan_crack_onset.py \
  --config sim/configs/crack_onset_scan.yaml \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 80 \
  --max-nonfinite-count 0
```
5. Record accepted set in `docs/parameter_register.md`:
- add date, commit hash, config path, and fit notes.

## Acceptance Criteria (Phase-2)
- `regress_phase2.py` passes.
- RuntimeWarning counts under configured thresholds.
- At least 1 crack-onset case from `scan_crack_onset.py`.
- Onset should be length-led: prefer `onset_length=true`; `onset_mean_aux` is auxiliary.
- Negative control (`no_notch_control`) should keep `onset=false` while `checks_ok=true`.
- Experimental overlay mismatch reduced and documented with plots/metrics.
- Mechanical branch should satisfy `max_mechanical_not_accepted_steps <= 160` under the current
  unilateral setup (`mech_unilateral_mode=volumetric`, `mech_preconditioner=jacobi`,
  `mech_clip_solution_on_limit=true`, `mech_regularization=1.0`).
- Crack branch should satisfy `max_crack_cg_nonconverged_steps <= 80` under the same scan setup.
