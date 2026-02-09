#!/usr/bin/env bash
set -euo pipefail

# Release Pack V2 command bundle
# Usage:
#   bash sim/tests/release_pack_v2_commands.sh
# Optional environment variables:
#   PYTHON=/path/to/python
#   DAY_TAG=2026-02-08

PYTHON_BIN="${PYTHON:-python}"
DAY_TAG="${DAY_TAG:-$(date +%F)}"
OUT_ROOT="sim/tests/regress_runs/${DAY_TAG}/release_pack_v2"

mkdir -p "${OUT_ROOT}"

echo "[1/5] phase2 full gate with exp + energy + D2 localization"
"${PYTHON_BIN}" sim/tests/regress_phase2.py \
  --out "${OUT_ROOT}/phase2_full" \
  --skip-phase1-suite \
  --scan-config sim/configs/crack_onset_scan.yaml \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 20 \
  --max-nonfinite-count 0 \
  --with-exp-alignment \
  --exp-alignment-config sim/configs/fatigue_lowamp_align_locked_v4.yaml \
  --with-energy-gate \
  --energy-gate-config sim/configs/fatigue_lowamp_align_locked_v4.yaml \
  --energy-gate-min-cycles 5 \
  --with-d2-localization \
  --d2-localization-config sim/configs/d2_localization_energy.yaml \
  --d2-min-cycles 3 \
  --d2-min-crack-delta 5.0e-2 \
  --d2-min-localization-index 3.0 \
  --d2-min-energy-crack-mean 1.0e-10 \
  --d2-min-energy-total-density-mean 1.0e-10 \
  --d2-max-runtime-warnings 50 \
  --d2-max-mechanical-not-accepted-steps 160 \
  --d2-max-crack-cg-nonconverged-steps 20 \
  --d2-max-nonfinite-count 0 \
  --d2-min-vtk-energy-fields 4

echo "[2/5] D3 multiphysics matrix gate"
"${PYTHON_BIN}" sim/tests/regress_d3_multiphysics_matrix.py \
  --config sim/configs/d3_multiphysics_matrix.yaml \
  --out "${OUT_ROOT}/d3_multiphysics_matrix/summary.json" \
  --require-all \
  --min-pass-count 3

echo "[3/5] full-case seed robustness batch 1"
"${PYTHON_BIN}" sim/tests/repeat_crack_onset_seeds.py \
  --base-config sim/configs/crack_onset_scan.yaml \
  --case-mode full \
  --seeds 41,42,43 \
  --out-root "${OUT_ROOT}/seed_batch_1" \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 20 \
  --max-nonfinite-count 0

echo "[4/5] full-case seed robustness batch 2"
"${PYTHON_BIN}" sim/tests/repeat_crack_onset_seeds.py \
  --base-config sim/configs/crack_onset_scan.yaml \
  --case-mode full \
  --seeds 44,45,46 \
  --out-root "${OUT_ROOT}/seed_batch_2" \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 20 \
  --max-nonfinite-count 0

echo "[5/5] multi-condition alignment full (5 conditions)"
"${PYTHON_BIN}" sim/tests/regress_exp_alignment_multi.py \
  --config sim/configs/exp_alignment_multi_d1_full.yaml \
  --out "${OUT_ROOT}/exp_alignment_multi/summary.json"

echo "[done] release pack v2 commands finished"

echo "[optional] week8 20-seed robustness template:"
echo "  ${PYTHON_BIN} sim/tests/run_seed_robustness_20.py --base-config sim/configs/crack_onset_scan.yaml --case-mode full --seed-start 41 --seed-count 20 --batch-size 5 --out-root ${OUT_ROOT}/seed_robustness_20"
