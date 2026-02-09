# Handoff: Release Pack V2

## Scope
This handoff packages the current Week 5-7 baseline into a reproducible command/report flow.

## Locked Inputs
- `sim/configs/release_pack_v2_lock.yaml`
- `sim/configs/crack_onset_scan.yaml`
- `sim/configs/fatigue_lowamp_align_locked_v4.yaml`
- `sim/configs/exp_alignment_multi_d1_full.yaml`
- `sim/configs/exp_alignment_multi_skeleton.yaml`

## Primary Commands
- One-shot command bundle:
  - `bash sim/tests/release_pack_v2_commands.sh`
  - `bash sim/tests/d1_full_gate_commands.sh`
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
- Units mapping in report:
  - 默认写入 `sigma_ref=168.4 GPa`、`L0=1e-6 m` 与 `rmse_tau/mae_tau` 的 `MPa <-> nd` 对照。
  - 可选覆盖参数：`--sigma-ref-gpa`、`--length-ref-m`、`--units-mapping-doc`、`--units-example-doc`。

## New Week 6/7 Components
- Energy-consistency gate:
  - `sim/tests/regress_energy_consistency.py`
- Phase2 integration flags:
  - `--with-energy-gate`
  - `--energy-gate-config`
  - `--energy-gate-min-cycles`
- Multi-condition alignment wrapper:
  - `sim/tests/regress_exp_alignment_multi.py`
  - Week-8 config example: `sim/configs/exp_alignment_multi_week8.yaml`
- Calibration sweep scaffold:
  - `sim/tests/sweep_calibration_multi.py`
  - Lock draft example: `sim/configs/fatigue_lowamp_align_lock_draft_week8.yaml`
- Seed robustness 20-batch template:
  - `sim/tests/run_seed_robustness_20.py`
- Seed CI aggregator:
  - `sim/tests/summarize_seed_robustness_ci.py`
  - Week-8 usage note: `docs/week8_seed_ci_and_artifact_rules_2026-02-08.md`

## Notes
- D1 full-gate now uses `sim/configs/exp_alignment_multi_d1_full.yaml` (5 enabled conditions).
- Skeleton config (`sim/configs/exp_alignment_multi_skeleton.yaml`) is kept for compatibility smoke only.
- Seed robustness in full-case mode is compute-heavy; keep outputs under `sim/tests/regress_runs/<date>/`.
