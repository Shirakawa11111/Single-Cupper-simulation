# Phase-1 Checklist

## D1 Baseline freeze
- [x] Run baseline regressions and store outputs under `sim/tests/regress_runs/2026-02-06/phase1_baseline/`.
- [x] Confirm all baseline tasks report `passed=true`.
- [x] Record run commands and paths in `docs/phase1_baseline_2026-02-06.md`.

## D2 Repository hygiene
- [x] Expand `.gitignore` for caches and generated simulation outputs.
- [x] Verify new outputs no longer pollute `git status`.

## D3 Reproducible environment
- [x] Add `requirements.txt`.
- [x] Add `requirements-dev.txt`.
- [x] Validate install in fresh virtual environment.

## D4 Documentation alignment
- [x] Fix README command drift (`regress_all` flags, seeded builder script path).
- [x] Add Phase-1 workflow pointers to README/HANDOFF.

## D5 Units and parameter ledger
- [x] Add `docs/units_mapping.md`.
- [x] Add `docs/parameter_register.md`.
- [x] Finalize `L0` and `burgers` physical convention for publication.

## D6 Config-driven execution
- [x] Add baseline YAML configs in `sim/configs/`.
- [x] Add `sim/tests/run_virtual_cycle_config.py`.
- [x] Dry-run and one actual run per config.

## D7 One-command suite
- [x] Add `sim/tests/run_phase1_suite.py`.
- [x] Run suite and archive `summary.json`.
- [x] Update `HANDOFF.md` with final paths and pass/fail snapshot.
