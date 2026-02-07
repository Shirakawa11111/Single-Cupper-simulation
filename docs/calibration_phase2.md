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
  --max-crack-cg-nonconverged-steps 20 \
  --max-nonfinite-count 0
```
3. Compare simulation curve (`virtual_cycle_stress_strain.csv`) with experiment:
- Match small-strain slope first.
- Then match loop width (`plastic_range`) and RSS peak trend.
4. Run crack-onset screening after each parameter update:
```bash
python sim/tests/scan_crack_onset.py \
  --config sim/configs/crack_onset_scan.yaml \
  --no-auto-output \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 20 \
  --min-notch-cycles-completed 3 \
  --max-nonfinite-count 0
```
5. Record accepted set in `docs/parameter_register.md`:
- add date, commit hash, config path, and fit notes.

### Optional: Week-3 DOE Pre-Screen
Use DOE pre-screen to narrow solver candidates before full `n=3` notch scan:
```bash
python sim/tests/sweep_crack_onset_doe.py \
  --base-config sim/configs/crack_onset_scan_quick.yaml \
  --tag doe_week3_quick \
  --max-runs 4 \
  --max-cases 1 \
  --min-notch-cycles-completed 1 \
  --mech-regularization-values 1.0,2.0 \
  --mech-solution-abs-limit-values 8,10 \
  --mech-accept-rel-residual-values 0.008
```
If full scan budgets are tight, use fast numerical pre-screen:
```bash
python sim/tests/sweep_crack_onset_doe.py \
  --base-config sim/configs/crack_onset_scan.yaml \
  --tag doe_week3_budget_case1_fast \
  --max-runs 2 \
  --max-cases 1 \
  --scan-timeout-s 180 \
  --vc-cycles 2 \
  --vc-cycle-points 20 \
  --vc-mech-max-iters 20 \
  --vc-mech-outer-max-iters 1 \
  --vc-mech-tol 2e-4 \
  --vc-mech-outer-tol 2e-5 \
  --min-notch-cycles-completed 1 \
  --mech-regularization-values 2.0,2.5 \
  --mech-solution-abs-limit-values 10 \
  --mech-accept-rel-residual-values 0.01
```
Then rescan shortlisted settings with:
- `--base-config sim/configs/crack_onset_scan.yaml`
- `--max-cases 3`
- `--min-notch-cycles-completed 3`

Practical note from 2026-02-07 DOE:
- `reg=2.5, limit=10` may look good on single-case clipping but can fail `notch_medium_drive` crack-CG checks.
- A better fast-budget bridge point was `reg=2.5, limit=8` (n=3 checks pass with lower clipping than `reg=2.0` baseline).
- However, full `n=3` verification still showed `notch_medium_drive` crack-CG collapse (`320` nonconverged steps), so crack-branch tuning remains required before lock.
- Crack-CG tuning outcome:
  - relaxing `crack_tol` alone did not resolve medium-case collapse.
  - increasing `crack_max_iters` to `1200` resolved medium-case crack-CG nonconvergence and restored full notch-3 gate pass.

## Acceptance Criteria (Phase-2)
- `regress_phase2.py` passes.
- RuntimeWarning counts under configured thresholds.
- At least 1 crack-onset case from `scan_crack_onset.py`.
- Onset should be length-led: prefer `onset_length=true`; `onset_mean_aux` is auxiliary.
- Negative control (`no_notch_control`) should keep `onset=false` while `checks_ok=true`.
- For trajectory-alignment runs, notch cases should preferably reach `cycles_completed >= 3`
  (current locked setup targets 4 cycles with `failure_threshold=0.999`).
- Experimental overlay mismatch reduced and documented with plots/metrics.
- Mechanical branch should satisfy `max_mechanical_not_accepted_steps <= 160` under the current
  unilateral setup (`mech_unilateral_mode=volumetric`, `mech_preconditioner=jacobi`,
  `mech_clip_solution_on_limit=true`, `mech_regularization=2.0`).
- Crack branch should satisfy `max_crack_cg_nonconverged_steps <= 20` under the same scan setup.
- Current trajectory-alignment lock uses `crack_length_threshold=0.995` and `failure_threshold=0.999`.
