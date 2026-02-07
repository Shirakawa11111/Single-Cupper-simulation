# Parameter Register (Phase-1)

## Material and fracture defaults
- Source: `sim/energy.py`
- `CopperParameters`
- `c11=1.0`, `c12=0.7209`, `c44=0.4477`
- `slip_resistance=1.3e-4`
- `hardening_modulus=1.0e-4`
- `hardening_b=8.0`
- `residual_stiffness=1e-6`
- `FractureParameters`
- `gc=1.0`, `l0=1.0`, `k=1e-6`, `epsilon_half=0.15`, `gres=0.1`
- `PFCParameters`
- `r=-0.25`, `u=0.25`, `q0=1.0`, `noise=1e-3`

## Coupling defaults
- Source: `sim/energy.py` `PFCCoupling(...)`
- `yield_tau=1.1e-4`
- `flow_scale=8.0e-4`
- `visco_exponent=3.0`
- `visco_ref=1.0e-4`
- `linear_hardening=4.0e-4`
- `kin_c=2.5e-4`
- `kin_d=1.2`
- `gamma0=1.0e-2`
- `slip_exponent=8.0`
- `h_iso=linear_hardening` (unless overridden)
- `h_gnd=0.0`

## Solver defaults
- Source: `sim/solver.py` `SolverConfig`
- `dt=1e-2`
- `plastic_relax=0.12`
- `crack_relax=1.0`
- `crack_eta=0.0`
- `crack_tol=1e-6`
- `crack_max_iters=400`
- `load_axis=0`
- `dir_coupling=0.3`
- `mech_plastic_weight=0.7`
- `pfc_active=True`
- `gnd_active=False`
- `gnd_burgers=1.0`

## Mechanical solver defaults
- Source: `sim/mechanics.py` `MechanicalConfig`
- `max_iters=200`
- `tol=1e-5`
- `unilateral=True`
- `unilateral_mode="spectral"`
- `outer_max_iters=5`
- `outer_tol=1e-6`

## Phase-1 baseline config set
- `sim/configs/monotonic_baseline.yaml`
- `sim/configs/fatigue_lowamp.yaml`
- `sim/configs/notch_gnd.yaml`

## Phase-2 / Week-4 locked config set
- `sim/configs/crack_onset_scan.yaml`
- `sim/configs/fatigue_lowamp_align_locked_v4.yaml`
- Week-4 release runner: `sim/tests/run_release_baseline_week4.py`
- Week-4 release report template: `docs/templates/week4_release_report_template.md`

## Frozen physical constants (Phase-1)
- `L0 = 1.0e-6 m` (code length unit to SI)
- `sigma_ref = 168.4 GPa`
- `b_phys = 2.556e-10 m`
- `gnd_burgers_nd (recommended for physical GND output) = b_phys / L0 = 2.556e-4`

## Reporting convention
- Regression thresholds and most current configs are still nd-first.
- If you report SI GND from run outputs, state:
- whether `gnd_burgers` was `1.0` (nd diagnostic mode) or `2.556e-4` (physical-reference mode)
- the conversion used (`docs/units_mapping.md`)

## Change-control rule
- Any parameter change affecting baseline behavior must include:
- updated config YAML
- updated regression summary under `sim/tests/regress_runs/<date>/...`
- one-line reason in `HANDOFF.md`
- If touching Week-4 locked configs, also update:
- `docs/week4_release_baseline_pack_2026-02-07.md`
- `WEEKLY_CHECKLIST.md`
