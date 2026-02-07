"""
Multi-condition experiment-alignment regression wrapper.

This is a lightweight orchestration layer that runs
`sim/tests/regress_exp_alignment.py` for each enabled condition in a YAML file
and aggregates one summary report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml  # type: ignore


def _default_out() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"exp_alignment_multi_{ts}" / "summary.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _str(val: Any) -> str:
    return str(val)


def _float(val: Any, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _int(val: Any, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return int(default)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-condition experiment-alignment gate.")
    parser.add_argument("--config", type=Path, default=Path("sim/configs/exp_alignment_multi_skeleton.yaml"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--only", type=str, default="", help="Optional comma-separated condition names to run.")
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")
    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    thresholds_default = defaults.get("thresholds", {})
    if not isinstance(thresholds_default, dict):
        thresholds_default = {}
    limits_default = defaults.get("limits", {})
    if not isinstance(limits_default, dict):
        limits_default = {}
    min_pass_count = _int(defaults.get("min_pass_count", 1), 1)

    rows = raw.get("conditions", [])
    if not isinstance(rows, list):
        raise ValueError("conditions must be a list.")

    selected = {name.strip() for name in args.only.split(",") if name.strip()}
    out_path = args.out or _default_out()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cond_dir = out_path.parent / "conditions"
    logs_dir = out_path.parent / "logs"
    cond_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now()
    records: list[dict[str, Any]] = []
    metrics_agg: dict[str, float] = {
        "rmse_tau_MPa_avg": 0.0,
        "mae_tau_MPa_avg": 0.0,
        "rmse_gamma_avg": 0.0,
        "mae_gamma_avg": 0.0,
    }
    metrics_count = 0

    for item in rows:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        if selected and name not in selected:
            continue
        if not bool(item.get("enabled", True)):
            continue

        thresholds = dict(thresholds_default)
        if isinstance(item.get("thresholds"), dict):
            thresholds.update(item.get("thresholds", {}))
        limits = dict(limits_default)
        if isinstance(item.get("limits"), dict):
            limits.update(item.get("limits", {}))

        cond_out = cond_dir / f"{name}.summary.json"
        cmd = [
            args.python,
            "sim/tests/regress_exp_alignment.py",
            "--config",
            _str(item.get("config", "sim/configs/fatigue_lowamp_align_locked_v4.yaml")),
            "--out",
            str(cond_out),
            "--rmse-tau-max",
            str(_float(thresholds.get("rmse_tau_max"), 30.0)),
            "--mae-tau-max",
            str(_float(thresholds.get("mae_tau_max"), 25.0)),
            "--rmse-gamma-max",
            str(_float(thresholds.get("rmse_gamma_max"), 4.2e-3)),
            "--max-runtime-warnings",
            str(_int(limits.get("max_runtime_warnings"), 50)),
            "--max-mechanical-not-accepted-steps",
            str(_int(limits.get("max_mechanical_not_accepted_steps"), 160)),
            "--max-crack-cg-nonconverged-steps",
            str(_int(limits.get("max_crack_cg_nonconverged_steps"), 40)),
            "--max-nonfinite-count",
            str(_int(limits.get("max_nonfinite_count"), 0)),
        ]

        # Condition-specific overrides for experiment/schmid mapping.
        optional_map = {
            "exp_folder": "--exp-folder",
            "cycle": "--cycle",
            "axis": "--axis",
            "stress_col": "--stress-col",
            "strain_col": "--strain-col",
            "exp_strain_scale": "--exp-strain-scale",
            "sim_stress_col": "--sim-stress-col",
            "sim_strain_col": "--sim-strain-col",
            "sim_stress_scale": "--sim-stress-scale",
        }
        for key, flag in optional_map.items():
            if key in item and item[key] is not None:
                val = _str(item[key])
                # Keep values like "-1,1,1" from being parsed as a new flag.
                if val.startswith("-"):
                    cmd.append(f"{flag}={val}")
                else:
                    cmd.extend([flag, val])

        t0 = perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = perf_counter() - t0

        stdout_path = logs_dir / f"{name}.stdout"
        stderr_path = logs_dir / f"{name}.stderr"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")

        summary = _read_json(cond_out)
        passed = bool(proc.returncode == 0 and isinstance(summary, dict) and summary.get("passed", False))
        rec = {
            "name": name,
            "command": cmd,
            "returncode": int(proc.returncode),
            "duration_s": dt,
            "passed": passed,
            "summary_json": str(cond_out),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "summary": summary,
        }
        records.append(rec)

        if isinstance(summary, dict) and isinstance(summary.get("metrics"), dict):
            m = summary["metrics"]
            metrics_agg["rmse_tau_MPa_avg"] += float(m.get("rmse_tau_MPa", 0.0))
            metrics_agg["mae_tau_MPa_avg"] += float(m.get("mae_tau_MPa", 0.0))
            metrics_agg["rmse_gamma_avg"] += float(m.get("rmse_gamma", 0.0))
            metrics_agg["mae_gamma_avg"] += float(m.get("mae_gamma", 0.0))
            metrics_count += 1

    if metrics_count > 0:
        for key in list(metrics_agg.keys()):
            metrics_agg[key] /= float(metrics_count)

    finished = datetime.now()
    total = len(records)
    passed_count = int(sum(1 for r in records if bool(r.get("passed", False))))
    failed_names = [str(r.get("name")) for r in records if not bool(r.get("passed", False))]
    overall_passed = total > 0 and passed_count >= min_pass_count and passed_count == total

    summary_payload = {
        "config": str(args.config),
        "out": str(out_path),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "duration_s": (finished - started).total_seconds(),
        "min_pass_count": min_pass_count,
        "condition_total": total,
        "passed_count": passed_count,
        "failed_names": failed_names,
        "passed": overall_passed,
        "metrics_average": metrics_agg if metrics_count > 0 else None,
        "conditions": records,
    }
    out_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
