"""
Small DOE sweep for experiment alignment on low-amplitude fatigue configs.

This script scans elastic constants (c11/c12/c44) and max_strain around
an existing locked config, then ranks candidates by rmse_tau_MPa.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml  # type: ignore


def _parse_floats(text: str) -> list[float]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


def _slug(v: float) -> str:
    s = f"{v:.6g}"
    return s.replace("-", "m").replace(".", "p")


def _default_out() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"exp_alignment_sweep_{ts}"


@dataclass
class Candidate:
    c11: float
    strain_scale: float


def _load_base(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    vc = data.get("virtual_cycle")
    if not isinstance(vc, dict):
        raise ValueError("Config must contain virtual_cycle mapping.")
    for k in ("c11", "c12", "c44", "max_strain"):
        if k not in vc:
            raise ValueError(f"Config virtual_cycle missing key: {k}")
    return data


def _write_candidate_config(
    base: dict[str, Any],
    out_path: Path,
    c11: float,
    c12_ratio: float,
    c44_ratio: float,
    strain_scale: float,
) -> None:
    data = json.loads(json.dumps(base))
    vc = data["virtual_cycle"]
    base_strain = float(vc["max_strain"])
    vc["c11"] = float(c11)
    vc["c12"] = float(c11 * c12_ratio)
    vc["c44"] = float(c11 * c44_ratio)
    vc["max_strain"] = float(base_strain * strain_scale)
    vc["task"] = f"{vc.get('task', 'align')}_c11{_slug(c11)}_ss{_slug(strain_scale)}"
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep experiment-alignment candidates.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("sim/configs/fatigue_lowamp_align_locked_v3.yaml"),
    )
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--c11-values", type=str, default="0.6,0.62,0.64")
    parser.add_argument("--strain-scale-values", type=str, default="0.98,1.0,1.02")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--rmse-tau-max", type=float, default=30.0)
    parser.add_argument("--mae-tau-max", type=float, default=25.0)
    parser.add_argument("--rmse-gamma-max", type=float, default=4.2e-3)
    parser.add_argument("--max-runtime-warnings", type=int, default=50)
    parser.add_argument("--max-mechanical-not-accepted-steps", type=int, default=160)
    parser.add_argument("--max-crack-cg-nonconverged-steps", type=int, default=20)
    parser.add_argument("--max-nonfinite-count", type=int, default=0)
    args = parser.parse_args()

    out_root = args.out_root or _default_out()
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_root = out_root / "candidates"
    cfg_root.mkdir(parents=True, exist_ok=True)

    base = _load_base(args.base_config)
    vc = base["virtual_cycle"]
    base_c11 = float(vc["c11"])
    c12_ratio = float(vc["c12"]) / base_c11
    c44_ratio = float(vc["c44"]) / base_c11
    base_strain = float(vc["max_strain"])

    c11_values = _parse_floats(args.c11_values)
    strain_scale_values = _parse_floats(args.strain_scale_values)

    candidates = [
        Candidate(c11=c11, strain_scale=ss)
        for c11, ss in itertools.product(c11_values, strain_scale_values)
    ]
    if args.max_runs > 0:
        candidates = candidates[: args.max_runs]

    rows: list[dict[str, Any]] = []
    t0 = perf_counter()
    for i, cand in enumerate(candidates, start=1):
        run_name = f"run_{i:03d}_c11{_slug(cand.c11)}_ss{_slug(cand.strain_scale)}"
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_root / f"{run_name}.yaml"
        _write_candidate_config(
            base=base,
            out_path=cfg_path,
            c11=cand.c11,
            c12_ratio=c12_ratio,
            c44_ratio=c44_ratio,
            strain_scale=cand.strain_scale,
        )
        sum_path = run_dir / "summary.json"
        cmd = [
            args.python,
            "sim/tests/regress_exp_alignment.py",
            "--config",
            str(cfg_path),
            "--out",
            str(sum_path),
            "--rmse-tau-max",
            str(args.rmse_tau_max),
            "--mae-tau-max",
            str(args.mae_tau_max),
            "--rmse-gamma-max",
            str(args.rmse_gamma_max),
            "--max-runtime-warnings",
            str(args.max_runtime_warnings),
            "--max-mechanical-not-accepted-steps",
            str(args.max_mechanical_not_accepted_steps),
            "--max-crack-cg-nonconverged-steps",
            str(args.max_crack_cg_nonconverged_steps),
            "--max-nonfinite-count",
            str(args.max_nonfinite_count),
        ]
        print(f"[run {i}/{len(candidates)}] {' '.join(cmd)}", flush=True)
        cp = subprocess.run(cmd)
        row: dict[str, Any] = {
            "run_name": run_name,
            "config_path": str(cfg_path),
            "summary_path": str(sum_path),
            "returncode": int(cp.returncode),
            "c11": cand.c11,
            "c12": cand.c11 * c12_ratio,
            "c44": cand.c11 * c44_ratio,
            "max_strain": base_strain * cand.strain_scale,
            "strain_scale": cand.strain_scale,
        }
        if sum_path.exists():
            data = json.loads(sum_path.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
            run = data.get("run", {})
            diag = run.get("stability_diagnostics", {}) if isinstance(run, dict) else {}
            row.update(
                {
                    "passed": bool(data.get("passed", False)),
                    "rmse_tau_MPa": metrics.get("rmse_tau_MPa"),
                    "mae_tau_MPa": metrics.get("mae_tau_MPa"),
                    "rmse_gamma": metrics.get("rmse_gamma"),
                    "mae_gamma": metrics.get("mae_gamma"),
                    "runtime_warning_count": run.get("runtime_warning_count"),
                    "mechanical_not_accepted_steps": diag.get("mechanical_not_accepted_steps"),
                    "crack_cg_nonconverged_steps": diag.get("crack_cg_nonconverged_steps"),
                    "nonfinite_count": diag.get("nonfinite_count"),
                }
            )
        rows.append(row)

    elapsed = perf_counter() - t0

    def _sort_key(r: dict[str, Any]) -> tuple[float, float, float]:
        rmse_tau = r.get("rmse_tau_MPa")
        rmse_gamma = r.get("rmse_gamma")
        mae_tau = r.get("mae_tau_MPa")
        return (
            float(rmse_tau) if rmse_tau is not None else 1e18,
            float(rmse_gamma) if rmse_gamma is not None else 1e18,
            float(mae_tau) if mae_tau is not None else 1e18,
        )

    ranked = sorted(rows, key=_sort_key)
    csv_path = out_root / "results.csv"
    fields: list[str] = []
    for r in ranked:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in ranked:
            w.writerow(r)

    summary = {
        "base_config": str(args.base_config),
        "out_root": str(out_root),
        "base_c11": base_c11,
        "base_max_strain": base_strain,
        "candidate_count": len(candidates),
        "duration_s": elapsed,
        "best": ranked[0] if ranked else None,
        "results_csv": str(csv_path),
    }
    sum_out = out_root / "summary.json"
    sum_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {sum_out}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
