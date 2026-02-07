# Week-4 Seed Robustness Round-1 (2026-02-07)

## Goal
- Verify robustness of:
  - notch case onset (`control_notch_mild`)
  - negative control non-onset (`no_notch_control`)
- Under multiple random seeds and multiple batches.

## Implementation
- `sim/tests/virtual_cycle.py` now exposes `random_seed` (was fixed at `42`).
- New runner: `sim/tests/repeat_crack_onset_seeds.py`.
- Each run keeps only two cases from `sim/configs/crack_onset_scan.yaml` for speed:
  - `control_notch_mild`
  - `no_notch_control`

## Batch Commands
```bash
python sim/tests/repeat_crack_onset_seeds.py \
  --base-config sim/configs/crack_onset_scan.yaml \
  --seeds 41,42,43 \
  --notch-case control_notch_mild \
  --negative-case no_notch_control \
  --out-root sim/tests/regress_runs/2026-02-07/crack_onset_seed_repeat_week4_n2_s41_43 \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 20 \
  --max-nonfinite-count 0
```

```bash
python sim/tests/repeat_crack_onset_seeds.py \
  --base-config sim/configs/crack_onset_scan.yaml \
  --seeds 44,45,46 \
  --notch-case control_notch_mild \
  --negative-case no_notch_control \
  --out-root sim/tests/regress_runs/2026-02-07/crack_onset_seed_repeat_week4_n2_s44_46 \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 20 \
  --max-nonfinite-count 0
```

## Results
- Batch A: `sim/tests/regress_runs/2026-02-07/crack_onset_seed_repeat_week4_n2_s41_43/summary.json`
  - `all_seed_gate_passed=true` (`3/3`)
- Batch B: `sim/tests/regress_runs/2026-02-07/crack_onset_seed_repeat_week4_n2_s44_46/summary.json`
  - `all_seed_gate_passed=true` (`3/3`)
- Combined:
  - total seeds: `6`
  - seed-gate pass: `6/6`
  - all seeds satisfy:
    - `notch_onset=true`
    - `negative_onset=false`
    - `checks_passed=true`
