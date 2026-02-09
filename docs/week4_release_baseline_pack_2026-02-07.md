# Week-4 Release Baseline Pack (2026-02-07)

## Scope
- Freeze a reproducible release bundle for the current Week-4 lock:
  - phase2 + crack-onset + exp-alignment gate（当前已默认集成 D2 局部化门禁）
  - D3 multiphysics matrix gate（默认开启）
  - seed robustness checks
  - report template for handoff/publication notes

## Locked Files
- `sim/configs/crack_onset_scan.yaml`
- `sim/configs/fatigue_lowamp_align_locked_v4.yaml`
- `sim/tests/regress_phase2.py` (default exp-alignment config -> v4)

## Bundle Runner
- Script: `sim/tests/run_release_baseline_week4.py`
- D3 matrix gate 默认开启；如需跳过可加 `--skip-d3-matrix`

### Quick profile (CI/dev smoke)
```bash
python sim/tests/run_release_baseline_week4.py \
  --profile quick \
  --out-root sim/tests/regress_runs/$(date +%F)/release_baseline_week4_quick
```

### Full scan without phase1 (recommended daily calibration loop)
```bash
python sim/tests/run_release_baseline_week4.py \
  --profile full_skip_phase1 \
  --out-root sim/tests/regress_runs/$(date +%F)/release_baseline_week4_full_skip_phase1
```

### Full release run with seed robustness
```bash
python sim/tests/run_release_baseline_week4.py \
  --profile full \
  --run-seed-robustness \
  --seed-batches "41,42,43;44,45,46" \
  --out-root sim/tests/regress_runs/$(date +%F)/release_baseline_week4_full
```

## Outputs
- Bundle summary: `<out-root>/bundle_summary.json`
- Task logs: `<out-root>/logs/*.stdout`, `<out-root>/logs/*.stderr`
- Phase2 summary: `<out-root>/phase2_gate/summary.json`
- D2 summary（默认开启）: `<out-root>/phase2_gate/d2_localization/summary.json`
- D3 summary（默认开启）: `<out-root>/d3_multiphysics_matrix/summary.json`
- Seed batches (if enabled): `<out-root>/seed_batch_*/summary.json`

## Report Template
- `docs/templates/week4_release_report_template.md`

## Current Week-4 Round-1 Evidence
- Alignment sweep report: `docs/week4_alignment_round1_2026-02-07.md`
- Seed robustness report: `docs/week4_seed_robustness_round1_2026-02-07.md`
