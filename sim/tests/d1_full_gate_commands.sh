#!/usr/bin/env bash
set -euo pipefail

# D1 full-gate command bundle
# Usage:
#   bash sim/tests/d1_full_gate_commands.sh
# Optional env:
#   PYTHON=/path/to/python
#   DAY_TAG=2026-02-09
#   WITH_PHASE1=1

PYTHON_BIN="${PYTHON:-python}"
DAY_TAG="${DAY_TAG:-$(date +%F)}"
OUT_ROOT="sim/tests/regress_runs/${DAY_TAG}/d1_full_gate"

CMD=(
  "${PYTHON_BIN}" sim/tests/run_d1_full_gate.py
  --out-root "${OUT_ROOT}"
  --run-seed-robustness
  --seed-case-mode full
)

if [[ "${WITH_PHASE1:-0}" == "1" ]]; then
  CMD+=(--with-phase1-suite)
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"
echo "[done] D1 full-gate finished: ${OUT_ROOT}/summary.json"
