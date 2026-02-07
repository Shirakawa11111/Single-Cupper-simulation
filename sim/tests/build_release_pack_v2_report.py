"""
Build a markdown report for release pack v2 from gate summaries.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _str(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build release pack v2 markdown report.")
    parser.add_argument("--phase2-summary", type=Path, default=None)
    parser.add_argument("--seed-batch1-summary", type=Path, default=None)
    parser.add_argument("--seed-batch2-summary", type=Path, default=None)
    parser.add_argument("--multi-align-summary", type=Path, default=None)
    parser.add_argument("--template", type=Path, default=Path("sim/tests/release_pack_v2_report_template.md"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--git-sha", type=str, default="unknown")
    args = parser.parse_args()

    template = args.template.read_text(encoding="utf-8")
    phase2 = _read_json(args.phase2_summary)
    batch1 = _read_json(args.seed_batch1_summary)
    batch2 = _read_json(args.seed_batch2_summary)
    multi = _read_json(args.multi_align_summary)

    b1_pass = bool(batch1 is not None and batch1.get("all_seed_gate_passed", False))
    b2_pass = bool(batch2 is not None and batch2.get("all_seed_gate_passed", False))
    seed_passed = b1_pass and b2_pass

    seed_pass_count = 0
    seed_total = 0
    for part in (batch1, batch2):
        if isinstance(part, dict):
            seed_pass_count += int(part.get("seed_gate_pass_count", 0))
            seed_total += int(part.get("seed_gate_total", 0))

    align_rmse_tau = None
    align_rmse_gamma = None
    if isinstance(multi, dict) and isinstance(multi.get("metrics_average"), dict):
        align_rmse_tau = multi["metrics_average"].get("rmse_tau_MPa_avg")
        align_rmse_gamma = multi["metrics_average"].get("rmse_gamma_avg")

    repl = {
        "{{DATE}}": date.today().isoformat(),
        "{{GIT_SHA}}": args.git_sha,
        "{{PHASE2_PASSED}}": _str(phase2.get("passed") if isinstance(phase2, dict) else None),
        "{{SEED_PASSED}}": _str(seed_passed if seed_total > 0 else None),
        "{{MULTI_ALIGN_PASSED}}": _str(multi.get("passed") if isinstance(multi, dict) else None),
        "{{PHASE2_WARNINGS}}": _str(phase2.get("total_runtime_warning_count") if isinstance(phase2, dict) else None),
        "{{SEED_PASS_COUNT}}": _str(seed_pass_count if seed_total > 0 else None),
        "{{SEED_TOTAL}}": _str(seed_total if seed_total > 0 else None),
        "{{ALIGN_RMSE_TAU}}": _str(align_rmse_tau),
        "{{ALIGN_RMSE_GAMMA}}": _str(align_rmse_gamma),
        "{{PHASE2_SUMMARY}}": _str(args.phase2_summary),
        "{{SEED_BATCH1_SUMMARY}}": _str(args.seed_batch1_summary),
        "{{SEED_BATCH2_SUMMARY}}": _str(args.seed_batch2_summary),
        "{{MULTI_ALIGN_SUMMARY}}": _str(args.multi_align_summary),
    }

    out_text = template
    for k, v in repl.items():
        out_text = out_text.replace(k, v)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out_text, encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

