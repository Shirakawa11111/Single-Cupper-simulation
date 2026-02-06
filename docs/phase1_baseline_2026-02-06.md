# Phase-1 Baseline (2026-02-06)

## Scope
- Baseline commit: `main@724231c`
- Goal: lock reproducible regression references before further Phase-1 refactors.

## Commands
```bash
python sim/tests/regress_microstrain.py --output sim/tests/regress_runs/2026-02-06/phase1_baseline/microstrain/summary.json
python sim/tests/regress_gnd.py --output sim/tests/regress_runs/2026-02-06/phase1_baseline/gnd/summary.json
python sim/tests/regress_gnd_cycle.py --output sim/tests/regress_runs/2026-02-06/phase1_baseline/gnd_cycle/summary.json
python sim/tests/regress_all.py --strict --log-dir sim/tests/regress_runs/2026-02-06/phase1_baseline/boundary_crack --output sim/tests/regress_runs/2026-02-06/phase1_baseline/boundary_crack/summary.json
```

## Baseline outputs
- `sim/tests/regress_runs/2026-02-06/phase1_baseline/microstrain/summary.json`
- `sim/tests/regress_runs/2026-02-06/phase1_baseline/gnd/summary.json`
- `sim/tests/regress_runs/2026-02-06/phase1_baseline/gnd_cycle/summary.json`
- `sim/tests/regress_runs/2026-02-06/phase1_baseline/boundary_crack/summary.json`

## Run snapshot
- Suite runner command:
```bash
python sim/tests/run_phase1_suite.py --strict --out sim/tests/regress_runs/2026-02-06/phase1_suite
```
- Aggregate summary: `sim/tests/regress_runs/2026-02-06/phase1_suite/summary.json`
- Result: `passed=true`
- Total wall time: `92.458 s`
- Subtask timings:
- `microstrain`: `0.381 s`
- `gnd`: `0.117 s`
- `gnd_cycle`: `53.984 s`
- `boundary_crack(strict)`: `37.974 s`

## Baseline freeze note
- `phase1_baseline` was synchronized from the strict suite outputs of `phase1_suite` on 2026-02-06.
- Use `phase1_suite/summary.json` as the primary audit trail; `phase1_baseline/` keeps the stable per-test JSON layout.

## Fresh-venv reproducibility check
- Command:
```bash
python3 -m venv .venv_phase1_check
source .venv_phase1_check/bin/activate
pip install -r requirements.txt
python sim/tests/run_phase1_suite.py --strict --out sim/tests/regress_runs/2026-02-06/phase1_suite_venvcheck
```
- Summary: `sim/tests/regress_runs/2026-02-06/phase1_suite_venvcheck/summary.json`
- Result: `passed=true` (total `79.387 s`)

## Config-run outputs
- `sim/tests/runs/2026-02-06/phase1_config_runs/monotonic_baseline_run_summary.json`
- `sim/tests/runs/2026-02-06/phase1_config_runs/fatigue_lowamp_run_summary.json`
- `sim/tests/runs/2026-02-06/phase1_config_runs/notch_gnd_run_summary.json`

## Notes
- Baseline artifacts are runtime outputs. They should stay under `sim/tests/regress_runs/`.
- Re-run this block after changing solver numerics, thresholds, or unit mapping assumptions.
