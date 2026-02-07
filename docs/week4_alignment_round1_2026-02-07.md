# Week-4 Alignment Round-1 (2026-02-07)

## Goal
- Reduce experiment-alignment stress mismatch (`rmse_tau_MPa`) without degrading
  strain mismatch (`rmse_gamma`) and without introducing solver instability.

## Command
```bash
python sim/tests/sweep_exp_alignment.py \
  --base-config sim/configs/fatigue_lowamp_align_locked_v3.yaml \
  --out-root sim/tests/regress_runs/2026-02-07/exp_alignment_sweep_week4_round1 \
  --c11-values 0.58,0.60,0.62 \
  --strain-scale-values 0.99,1.0 \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 20 \
  --max-nonfinite-count 0
```

## Result Table
Source: `sim/tests/regress_runs/2026-02-07/exp_alignment_sweep_week4_round1/results.csv`

| rank | c11 | strain_scale | rmse_tau_MPa | mae_tau_MPa | rmse_gamma | passed |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.58 | 0.99 | 28.447 | 23.181 | 3.889e-3 | true |
| 2 | 0.58 | 1.00 | 28.565 | 23.270 | 3.909e-3 | true |
| 3 | 0.60 | 0.99 | 28.851 | 23.493 | 3.889e-3 | true |
| 4 | 0.60 | 1.00 | 28.974 | 23.589 | 3.909e-3 | true |
| 5 | 0.62 | 0.99 | 29.261 | 23.812 | 3.889e-3 | true |
| 6 | 0.62 | 1.00 | 29.391 | 23.912 | 3.909e-3 | true |

All candidates kept:
- `runtime_warning_count=0`
- `mechanical_not_accepted_steps=0`
- `crack_cg_nonconverged_steps=0`
- `nonfinite_count=0`

## Selected Lock (v4)
- Config: `sim/configs/fatigue_lowamp_align_locked_v4.yaml`
- Key params:
  - `c11=0.58`
  - `c12=0.418122`
  - `c44=0.259666`
  - `max_strain=0.00127324197` (`0.99 × v3`)
- Verification:
  - `sim/tests/regress_runs/2026-02-07/exp_alignment_gate_v4/summary.json`
  - `rmse_tau_MPa=28.447`
  - `mae_tau_MPa=23.181`
  - `rmse_gamma=3.889e-3`

## Chain Check
- `sim/tests/regress_runs/2026-02-07/phase2_gate_with_exp_v4_skip_phase1/summary.json`
- `passed=true`
