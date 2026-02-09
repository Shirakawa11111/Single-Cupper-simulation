"""
Sweep calibration parameters against multi-condition alignment gate.

Workflow:
1) Build candidate simulation configs from a base config + parameter grid.
2) Run multi-condition alignment gate for each candidate.
3) Rank candidates and write best-config / lock-draft outputs.
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


def _default_out_root() -> Path:
    day = date.today().isoformat()
    ts = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/regress_runs") / day / f"calibration_multi_{ts}"


def _parse_float_list(text: str | None, default: list[float]) -> list[float]:
    if text is None or not text.strip():
        return list(default)
    vals: list[float] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    return vals if vals else list(default)


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return raw


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@dataclass
class CandidateResult:
    idx: int
    params: dict[str, float]
    passed: bool
    passed_count: int
    condition_total: int
    rmse_tau_avg: float
    mae_tau_avg: float
    rmse_gamma_avg: float
    mae_gamma_avg: float
    duration_s: float
    returncode: int
    summary_json: str
    sim_config: str
    multi_config: str


def _metrics_from_summary(summary: dict[str, Any] | None) -> tuple[bool, int, int, float, float, float, float]:
    if not isinstance(summary, dict):
        return False, 0, 0, float("inf"), float("inf"), float("inf"), float("inf")
    passed = bool(summary.get("passed", False))
    passed_count = int(summary.get("passed_count", 0))
    condition_total = int(summary.get("condition_total", 0))
    m = summary.get("metrics_average")
    if not isinstance(m, dict):
        return passed, passed_count, condition_total, float("inf"), float("inf"), float("inf"), float("inf")
    return (
        passed,
        passed_count,
        condition_total,
        float(m.get("rmse_tau_MPa_avg", float("inf"))),
        float(m.get("mae_tau_MPa_avg", float("inf"))),
        float(m.get("rmse_gamma_avg", float("inf"))),
        float(m.get("mae_gamma_avg", float("inf"))),
    )


def _write_lock_draft(path: Path, sim_cfg: dict[str, Any], source_summary: str, params: dict[str, float]) -> None:
    payload = dict(sim_cfg)
    vc = payload.get("virtual_cycle")
    if isinstance(vc, dict):
        for key in ("csv_output", "analysis_csv", "data_output", "dump_dir", "vtk_dir", "initial_vtk", "stress_strain_csv"):
            vc.pop(key, None)
    payload["description"] = (
        "Week-8 lock draft from sweep_calibration_multi.py "
        f"(source_summary={source_summary})."
    )
    payload["calibration_meta"] = {
        "source_summary": source_summary,
        "selected_params": params,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep multi-condition calibration candidates.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--base-sim-config", type=Path, default=Path("sim/configs/fatigue_lowamp_align_locked_v4.yaml"))
    parser.add_argument("--multi-config", type=Path, default=Path("sim/configs/exp_alignment_multi_week8.yaml"))
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-strain-values", type=str, default=None, help="Comma-separated floats.")
    parser.add_argument("--c11-values", type=str, default=None, help="Comma-separated floats.")
    parser.add_argument("--yield-tau-values", type=str, default=None, help="Comma-separated floats.")
    parser.add_argument("--flow-scale-values", type=str, default=None, help="Comma-separated floats.")
    parser.add_argument("--linear-hardening-values", type=str, default=None, help="Comma-separated floats.")
    parser.add_argument("--best-config-out", type=Path, default=None)
    parser.add_argument("--lock-draft-out", type=Path, default=None)
    args = parser.parse_args()

    out_root = args.out_root or _default_out_root()
    out_root.mkdir(parents=True, exist_ok=True)
    candidates_dir = out_root / "candidates"
    logs_dir = out_root / "logs"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    sim_base = _read_yaml(args.base_sim_config)
    vc = sim_base.get("virtual_cycle")
    if not isinstance(vc, dict):
        raise ValueError("base sim config must contain virtual_cycle mapping")

    multi_base = _read_yaml(args.multi_config)
    conds = multi_base.get("conditions")
    if not isinstance(conds, list):
        raise ValueError("multi config must contain conditions list")

    default_max_strain = [float(vc.get("max_strain", 0.00127324197))]
    default_c11 = [float(vc.get("c11", 0.58))]
    default_yield_tau = [float(vc.get("yield_tau", 0.04))]
    default_flow_scale = [float(vc.get("flow_scale", 4.0))]
    default_linear_hardening = [float(vc.get("linear_hardening", 0.0))]

    max_strain_values = _parse_float_list(args.max_strain_values, default_max_strain)
    c11_values = _parse_float_list(args.c11_values, default_c11)
    yield_tau_values = _parse_float_list(args.yield_tau_values, default_yield_tau)
    flow_scale_values = _parse_float_list(args.flow_scale_values, default_flow_scale)
    linear_hardening_values = _parse_float_list(args.linear_hardening_values, default_linear_hardening)

    grid = list(
        itertools.product(
            max_strain_values,
            c11_values,
            yield_tau_values,
            flow_scale_values,
            linear_hardening_values,
        )
    )
    if args.max_runs is not None:
        grid = grid[: max(args.max_runs, 0)]
    if not grid:
        raise ValueError("No candidate grid to run.")

    results: list[CandidateResult] = []
    for idx, (max_strain, c11, yield_tau, flow_scale, linear_hardening) in enumerate(grid, start=1):
        tag = f"cand_{idx:03d}"
        cand_dir = candidates_dir / tag
        cand_dir.mkdir(parents=True, exist_ok=True)
        sim_cfg = json.loads(json.dumps(sim_base))
        sim_vc = sim_cfg["virtual_cycle"]
        sim_vc["task"] = f"calib_{tag}"
        sim_vc["max_strain"] = float(max_strain)
        sim_vc["c11"] = float(c11)
        sim_vc["yield_tau"] = float(yield_tau)
        sim_vc["flow_scale"] = float(flow_scale)
        sim_vc["linear_hardening"] = float(linear_hardening)
        sim_cfg_path = cand_dir / "sim_config.yaml"
        sim_cfg_path.write_text(yaml.safe_dump(sim_cfg, sort_keys=False), encoding="utf-8")

        multi_cfg = json.loads(json.dumps(multi_base))
        defaults = multi_cfg.setdefault("defaults", {})
        if isinstance(defaults, dict):
            defaults["reuse_first_sim_csv"] = True
        for cond in multi_cfg["conditions"]:
            if isinstance(cond, dict) and bool(cond.get("enabled", True)):
                cond["config"] = str(sim_cfg_path)
        multi_cfg_path = cand_dir / "multi_config.yaml"
        multi_cfg_path.write_text(yaml.safe_dump(multi_cfg, sort_keys=False), encoding="utf-8")

        summary_path = cand_dir / "summary.json"
        cmd = [
            args.python,
            "sim/tests/regress_exp_alignment_multi.py",
            "--config",
            str(multi_cfg_path),
            "--out",
            str(summary_path),
        ]
        t0 = perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = perf_counter() - t0
        (logs_dir / f"{tag}.stdout").write_text(proc.stdout, encoding="utf-8")
        (logs_dir / f"{tag}.stderr").write_text(proc.stderr, encoding="utf-8")

        summary = _read_json(summary_path)
        passed, passed_count, condition_total, rmse_tau, mae_tau, rmse_gamma, mae_gamma = _metrics_from_summary(
            summary
        )
        res = CandidateResult(
            idx=idx,
            params={
                "max_strain": float(max_strain),
                "c11": float(c11),
                "yield_tau": float(yield_tau),
                "flow_scale": float(flow_scale),
                "linear_hardening": float(linear_hardening),
            },
            passed=bool(proc.returncode == 0 and passed),
            passed_count=passed_count,
            condition_total=condition_total,
            rmse_tau_avg=rmse_tau,
            mae_tau_avg=mae_tau,
            rmse_gamma_avg=rmse_gamma,
            mae_gamma_avg=mae_gamma,
            duration_s=dt,
            returncode=int(proc.returncode),
            summary_json=str(summary_path),
            sim_config=str(sim_cfg_path),
            multi_config=str(multi_cfg_path),
        )
        results.append(res)
        print(
            f"[cand {idx}] passed={res.passed} pass_count={res.passed_count}/{res.condition_total} "
            f"rmse_tau={res.rmse_tau_avg:.6f} rmse_gamma={res.rmse_gamma_avg:.6e}"
        )

    ranked = sorted(
        results,
        key=lambda r: (
            0 if r.passed else 1,
            r.rmse_tau_avg,
            r.rmse_gamma_avg,
            r.duration_s,
            r.idx,
        ),
    )

    csv_path = out_root / "runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "idx",
                "passed",
                "passed_count",
                "condition_total",
                "rmse_tau_avg",
                "mae_tau_avg",
                "rmse_gamma_avg",
                "mae_gamma_avg",
                "duration_s",
                "max_strain",
                "c11",
                "yield_tau",
                "flow_scale",
                "linear_hardening",
                "summary_json",
                "sim_config",
                "multi_config",
            ],
        )
        writer.writeheader()
        for rank, r in enumerate(ranked, start=1):
            row = {
                "rank": rank,
                "idx": r.idx,
                "passed": r.passed,
                "passed_count": r.passed_count,
                "condition_total": r.condition_total,
                "rmse_tau_avg": r.rmse_tau_avg,
                "mae_tau_avg": r.mae_tau_avg,
                "rmse_gamma_avg": r.rmse_gamma_avg,
                "mae_gamma_avg": r.mae_gamma_avg,
                "duration_s": r.duration_s,
                "summary_json": r.summary_json,
                "sim_config": r.sim_config,
                "multi_config": r.multi_config,
            }
            row.update(r.params)
            writer.writerow(row)

    best = ranked[0]
    best_config_out = args.best_config_out or (out_root / "best_sim_config.yaml")
    best_config_out.parent.mkdir(parents=True, exist_ok=True)
    best_config_out.write_text(Path(best.sim_config).read_text(encoding="utf-8"), encoding="utf-8")

    if args.lock_draft_out is not None:
        _write_lock_draft(
            args.lock_draft_out,
            _read_yaml(Path(best.sim_config)),
            best.summary_json,
            best.params,
        )

    summary_payload = {
        "base_sim_config": str(args.base_sim_config),
        "multi_config": str(args.multi_config),
        "out_root": str(out_root),
        "candidate_count": len(results),
        "max_runs": args.max_runs,
        "runs_csv": str(csv_path),
        "best_rank_idx": best.idx,
        "best_passed": best.passed,
        "best_params": best.params,
        "best_summary_json": best.summary_json,
        "best_config_out": str(best_config_out),
        "lock_draft_out": str(args.lock_draft_out) if args.lock_draft_out else None,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"[done] {summary_path}")
    print(json.dumps(summary_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
