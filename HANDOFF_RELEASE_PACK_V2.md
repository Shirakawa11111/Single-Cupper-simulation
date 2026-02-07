# Handoff: Release Pack V2

## Scope
This handoff packages the current Week 5-7 baseline into a reproducible command/report flow.

## Locked Inputs
- `sim/configs/release_pack_v2_lock.yaml`
- `sim/configs/crack_onset_scan.yaml`
- `sim/configs/fatigue_lowamp_align_locked_v4.yaml`
- `sim/configs/exp_alignment_multi_skeleton.yaml`

## Primary Commands
- One-shot command bundle:
  - `bash sim/tests/release_pack_v2_commands.sh`
- Individual components:
  - `python sim/tests/regress_phase2.py ... --with-exp-alignment --with-energy-gate`
  - `python sim/tests/repeat_crack_onset_seeds.py --case-mode full --seeds 41,42,43 ...`
  - `python sim/tests/repeat_crack_onset_seeds.py --case-mode full --seeds 44,45,46 ...`
  - `python sim/tests/regress_exp_alignment_multi.py --config sim/configs/exp_alignment_multi_skeleton.yaml ...`

## Reporting
- Template:
  - `sim/tests/release_pack_v2_report_template.md`
- Generator:
  - `python sim/tests/build_release_pack_v2_report.py --phase2-summary ... --seed-batch1-summary ... --seed-batch2-summary ... --multi-align-summary ... --out ...`

## New Week 6/7 Components
- Energy-consistency gate:
  - `sim/tests/regress_energy_consistency.py`
- Phase2 integration flags:
  - `--with-energy-gate`
  - `--energy-gate-config`
  - `--energy-gate-min-cycles`
- Multi-condition alignment wrapper:
  - `sim/tests/regress_exp_alignment_multi.py`

## Notes
- Current multi-condition config is a skeleton with one enabled condition.
- Seed robustness in full-case mode is compute-heavy; keep outputs under `sim/tests/regress_runs/<date>/`.
