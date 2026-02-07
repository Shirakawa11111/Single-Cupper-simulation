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

echo "[1/4] phase2 full gate with exp + energy"
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
  --energy-gate-min-cycles 5

echo "[2/4] full-case seed robustness batch 1"
"${PYTHON_BIN}" sim/tests/repeat_crack_onset_seeds.py \
  --base-config sim/configs/crack_onset_scan.yaml \
  --case-mode full \
  --seeds 41,42,43 \
  --out-root "${OUT_ROOT}/seed_batch_1" \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 20 \
  --max-nonfinite-count 0

echo "[3/4] full-case seed robustness batch 2"
"${PYTHON_BIN}" sim/tests/repeat_crack_onset_seeds.py \
  --base-config sim/configs/crack_onset_scan.yaml \
  --case-mode full \
  --seeds 44,45,46 \
  --out-root "${OUT_ROOT}/seed_batch_2" \
  --max-runtime-warnings 50 \
  --max-mechanical-not-accepted-steps 160 \
  --max-crack-cg-nonconverged-steps 20 \
  --max-nonfinite-count 0

echo "[4/4] multi-condition alignment skeleton (current single condition)"
"${PYTHON_BIN}" sim/tests/regress_exp_alignment_multi.py \
  --config sim/configs/exp_alignment_multi_skeleton.yaml \
  --out "${OUT_ROOT}/exp_alignment_multi/summary.json"

echo "[done] release pack v2 commands finished"
