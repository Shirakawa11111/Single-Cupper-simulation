"""
Repeat crack-onset scan across random seeds for robustness checks.

Focus criterion:
- notch case should onset
- negative control should remain non-onset
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml  # type: ignore


def _parse_ints(text: str) -> list[int]:
    vals: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("Expected at least one integer.")
    return vals


def _default_out() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"crack_onset_seed_repeat_{ts}"


def _deepcopy_yaml_obj(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def _extract_case(summary: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in summary.get("cases", []):
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return None


def _extract_all_cases(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("cases", [])
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat crack-onset scans over random seeds.")
    parser.add_argument("--base-config", type=Path, default=Path("sim/configs/crack_onset_scan.yaml"))
    parser.add_argument("--seeds", type=str, default="41,42,43")
    parser.add_argument(
        "--case-mode",
        type=str,
        choices=("pair", "full"),
        default="pair",
        help="pair: only notch/negative selected cases; full: all cases from config.",
    )
    parser.add_argument("--notch-case", type=str, default="control_notch_mild")
    parser.add_argument("--negative-case", type=str, default="no_notch_control")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--auto-output", action="store_true", help="Enable per-case VTK/LAMMPS output.")
    parser.add_argument("--max-runtime-warnings", type=int, default=50)
    parser.add_argument("--max-mechanical-not-accepted-steps", type=int, default=160)
    parser.add_argument("--max-crack-cg-nonconverged-steps", type=int, default=20)
    parser.add_argument("--max-nonfinite-count", type=int, default=0)
    args = parser.parse_args()

    seeds = _parse_ints(args.seeds)
    out_root = args.out_root or _default_out()
    out_root.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Base config root must be a mapping.")
    defaults = raw.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("Base config defaults must be a mapping.")
    vc = defaults.get("virtual_cycle")
    if not isinstance(vc, dict):
        raise ValueError("Base config defaults.virtual_cycle must be a mapping.")
    cases = raw.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Base config cases must be a list.")

    selected_cases: list[dict[str, Any]]
    if args.case_mode == "pair":
        # Keep only the two target cases for faster robustness checks.
        keep = {args.notch_case, args.negative_case}
        selected_cases = [c for c in cases if isinstance(c, dict) and c.get("name") in keep]
        if len(selected_cases) != 2:
            raise ValueError(
                f"Expected to find exactly 2 selected cases ({args.notch_case}, {args.negative_case})."
            )
    else:
        selected_cases = [c for c in cases if isinstance(c, dict)]
        if not selected_cases:
            raise ValueError("No valid cases found in base config.")

    rows: list[dict[str, Any]] = []
    t0 = perf_counter()
    for seed in seeds:
        run_dir = out_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = _deepcopy_yaml_obj(raw)
        cfg["cases"] = _deepcopy_yaml_obj(selected_cases)
        cfg["defaults"]["virtual_cycle"]["random_seed"] = int(seed)
        cfg_path = run_dir / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        scan_out = run_dir / "scan"
        cmd = [
            args.python,
            "sim/tests/scan_crack_onset.py",
            "--config",
            str(cfg_path),
            "--out",
            str(scan_out),
            "--max-runtime-warnings",
            str(args.max_runtime_warnings),
            "--max-mechanical-not-accepted-steps",
            str(args.max_mechanical_not_accepted_steps),
            "--max-crack-cg-nonconverged-steps",
            str(args.max_crack_cg_nonconverged_steps),
            "--max-nonfinite-count",
            str(args.max_nonfinite_count),
        ]
        if args.auto_output:
            cmd.append("--auto-output")
        else:
            cmd.append("--no-auto-output")
        print(f"[seed {seed}] {' '.join(cmd)}", flush=True)
        cp = subprocess.run(cmd)

        row: dict[str, Any] = {
            "seed": seed,
            "config_path": str(cfg_path),
            "scan_out": str(scan_out),
            "returncode": int(cp.returncode),
        }
        summary_path = scan_out / "summary.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            passed = bool(data.get("passed", False))
            checks_passed = bool(data.get("checks_passed", False))
            row.update(
                {
                    "passed": passed,
                    "checks_passed": checks_passed,
                    "onset_cases": data.get("onset_cases"),
                    "cases_total": data.get("cases_total"),
                }
            )
            if args.case_mode == "pair":
                notch = _extract_case(data, args.notch_case) or {}
                neg = _extract_case(data, args.negative_case) or {}
                row.update(
                    {
                        "notch_onset": notch.get("onset"),
                        "notch_onset_length": notch.get("onset_length"),
                        "notch_checks_ok": notch.get("checks_ok"),
                        "notch_cycles_completed": notch.get("cycles_completed"),
                        "negative_onset": neg.get("onset"),
                        "negative_checks_ok": neg.get("checks_ok"),
                        "negative_cycles_completed": neg.get("cycles_completed"),
                    }
                )
                row["seed_gate_pass"] = bool(
                    passed
                    and checks_passed
                    and row.get("notch_onset")
                    and row.get("notch_checks_ok")
                    and (row.get("negative_onset") is False)
                    and row.get("negative_checks_ok")
                )
            else:
                all_cases = _extract_all_cases(data)
                notch_rows = [r for r in all_cases if bool(r.get("notch_case", False))]
                neg_rows = [r for r in all_cases if not bool(r.get("notch_case", False))]
                notch_onset_all = bool(notch_rows) and all(r.get("onset") is True for r in notch_rows)
                notch_checks_all = bool(notch_rows) and all(bool(r.get("checks_ok")) for r in notch_rows)
                neg_onset_all_false = bool(neg_rows) and all(r.get("onset") is False for r in neg_rows)
                neg_checks_all = bool(neg_rows) and all(bool(r.get("checks_ok")) for r in neg_rows)
                notch_cycles_min = min(int(r.get("cycles_completed", 0)) for r in notch_rows) if notch_rows else None
                neg_cycles_min = min(int(r.get("cycles_completed", 0)) for r in neg_rows) if neg_rows else None
                row.update(
                    {
                        "notch_case_count": len(notch_rows),
                        "negative_case_count": len(neg_rows),
                        "notch_onset_all": notch_onset_all,
                        "notch_checks_all": notch_checks_all,
                        "notch_cycles_min": notch_cycles_min,
                        "negative_onset_all_false": neg_onset_all_false,
                        "negative_checks_all": neg_checks_all,
                        "negative_cycles_min": neg_cycles_min,
                    }
                )
                row["seed_gate_pass"] = bool(
                    passed
                    and checks_passed
                    and notch_onset_all
                    and notch_checks_all
                    and neg_onset_all_false
                    and neg_checks_all
                )
        rows.append(row)

    elapsed = perf_counter() - t0
    rows_sorted = sorted(rows, key=lambda r: int(r.get("seed", 0)))
    csv_path = out_root / "results.csv"
    fields: list[str] = []
    for r in rows_sorted:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)

    seed_gate_passes = [bool(r.get("seed_gate_pass", False)) for r in rows_sorted]
    summary = {
        "base_config": str(args.base_config),
        "out_root": str(out_root),
        "seeds": seeds,
        "case_mode": args.case_mode,
        "notch_case": args.notch_case,
        "negative_case": args.negative_case,
        "duration_s": elapsed,
        "seed_gate_pass_count": int(sum(1 for p in seed_gate_passes if p)),
        "seed_gate_total": len(seed_gate_passes),
        "all_seed_gate_passed": bool(seed_gate_passes) and all(seed_gate_passes),
        "results_csv": str(csv_path),
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
