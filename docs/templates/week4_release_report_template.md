# Week-4 Release Report Template

## Metadata
- Date:
- Commit:
- Runner profile (`quick` / `full_skip_phase1` / `full`):
- Bundle summary path:

## Locked Configs
- Phase-2 scan config: `sim/configs/crack_onset_scan.yaml`
- Exp-alignment config: `sim/configs/fatigue_lowamp_align_locked_v4.yaml`

## Gate Results
- `phase2_gate.passed`:
- `phase2_gate.total_runtime_warning_count`:
- `phase2_gate.onset_cases`:
- `phase2_gate.exp_alignment.rmse_tau_MPa`:
- `phase2_gate.exp_alignment.rmse_gamma`:

## Seed Robustness
- Batch A summary path:
- Batch B summary path:
- Combined `seed_gate_pass / total`:

## Key Metrics Snapshot
| metric | value |
|---|---:|
| notch clipping total |  |
| mechanical_not_accepted_steps (max) |  |
| crack_cg_nonconverged_steps (max) |  |
| rmse_tau_MPa |  |
| mae_tau_MPa |  |
| rmse_gamma |  |

## Units Mapping Snapshot (nd↔SI)
- Reference: `sigma_ref = 168.4 GPa (=168400 MPa)`, `L0 = 1.0e-6 m`
- Stress conversion: `tau_MPa = tau_nd * 168400`, `tau_nd = tau_MPa / 168400`
- Fill-in:
  - `rmse_tau_nd`:
  - `mae_tau_nd`:
- Worked example: `docs/week9_units_example_2026-02-08.md`

## Notes
- Numerical stability notes:
- Physical interpretation notes:
- Follow-up tasks:
