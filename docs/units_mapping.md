# Units Mapping

## Current model convention
- The core solver is non-dimensional.
- Reference statement: `sim/energy.py` documents `dx=1`, `sigma_ref=c11`, and non-dimensional free-energy scaling.

## Phase-1 frozen physical reference (2026-02-06)
- `L0 = 1.0e-6 m` (one code length unit maps to one micron).
- `sigma_ref = 168.4 GPa` (Cu `c11` reference used in code).
- `b_phys = 2.556e-10 m` (Cu FCC Burgers vector magnitude, `a/sqrt(2)` with `a≈3.615e-10 m`).
- `gnd_burgers_nd_recommended = b_phys / L0 = 2.556e-4`.

## Recommended Phase-1 mapping table
- `length_nd -> length_phys`: `x_phys = x_nd * L0`
- `stress_nd -> stress_phys`: `sigma_phys = sigma_nd * sigma_ref`
- `strain_nd -> strain_phys`: identical for small strain (`epsilon_phys = epsilon_nd`)
- `time_nd -> time_phys`: define by calibration to one dynamic reference test (`t_phys = t_nd * t0`)
- `energy_nd -> energy_phys`: `E_phys = E_nd * sigma_ref * L0^3`

## Default reference choices (frozen for Phase-1)
- `sigma_ref = 168.4 GPa` (Cu `c11`, see `CopperParameters` description in `sim/energy.py`).
- `L0 = 1.0e-6 m`.
- `b = 2.556e-10 m`.
- For physical GND reporting, set `gnd_burgers = 2.556e-4` in config/CLI.

## GND interpretation
- Code output `gnd_density` currently comes from `||alpha||_F / burgers`.
- If `L0` is set to meters and `burgers` uses meters, `gnd_density` maps to `m^-2`.
- Generic conversion from code output to SI:
- `rho_phys = rho_code * gnd_burgers_nd / (L0 * b_phys)`.
- If `gnd_burgers_nd = b_phys / L0` (recommended), this simplifies to:
- `rho_phys = rho_code / L0^2`.

## Worked example (Week-9)
### Example A: GND output row (`nd -> SI`)
- Source: `sim/tests/regress_runs/2026-02-06/phase1_suite_venvcheck/gnd_cycle/summary.json`
- Input field: `results.gnd_means[4] = 2.6317655902047965e-05` (code nd).
- Frozen references:
  - `L0 = 1.0e-6 m`
  - `b_phys = 2.556e-10 m`
  - `gnd_burgers_nd = 2.556e-4 = b_phys / L0`
- Conversion:
  - General: `rho_phys = rho_code * gnd_burgers_nd / (L0 * b_phys)`
  - Simplified (recommended): `rho_phys = rho_code / L0^2`
- Computed output:
  - `rho_phys = 2.6317655902047965e-05 / (1.0e-6)^2 = 2.6317655902047965e+07 m^-2`

### Example B: alignment stress metric (`MPa <-> nd`)
- Source: `sim/tests/regress_runs/2026-02-08/exp_alignment_multi_week8_smoke/conditions/lowamp_cycle_1000.summary.json`
- Input field: `metrics.rmse_tau_MPa = 28.44743777411884 MPa`.
- Reference: `sigma_ref = 168.4 GPa = 168400 MPa`.
- Conversion:
  - `tau_MPa = tau_nd * sigma_ref_MPa`
  - `tau_nd = tau_MPa / sigma_ref_MPa`
- Computed output:
  - `rmse_tau_nd = 28.44743777411884 / 168400 = 1.689277777560501e-04`

## Phase-1 action items
- [x] Freeze one `L0` convention and publish it in `docs/parameter_register.md`.
- [x] Freeze one `b` convention and corresponding `gnd_burgers_nd`.
- [x] Add one worked example converting a regression output row from nd to SI (`docs/week9_units_example_2026-02-08.md`).
- [x] Keep conversion logic in analysis scripts, not hidden inside solver kernels.
