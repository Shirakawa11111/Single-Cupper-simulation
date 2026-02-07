"""
Week-3 DOE helper for crack-onset scan tuning.

Builds a Cartesian parameter matrix, runs `scan_crack_onset.py` per candidate,
and writes run/case leaderboards for quick screening.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml  # type: ignore

ROOT = Path(__file__).resolve().parents[2]


def _default_out_dir(tag: str) -> Path:
    return Path("sim/tests/regress_runs") / date.today().isoformat() / tag


def _parse_floats(text: str) -> list[float]:
    vals: list[float] = []
    for token in text.split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


def _safe_id(value: float) -> str:
    s = f"{value:.6g}"
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be mapping: {path}")
    return raw


def _ensure_defaults(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults = cfg.setdefault("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be mapping.")
    vc = defaults.setdefault("virtual_cycle", {})
    if not isinstance(vc, dict):
        raise ValueError("defaults.virtual_cycle must be mapping.")
    criteria = defaults.setdefault("criteria", {})
    if not isinstance(criteria, dict):
        raise ValueError("defaults.criteria must be mapping.")
    return vc, criteria


@dataclass(frozen=True)
class Candidate:
    mech_regularization: float
    mech_solution_abs_limit: float
    mech_accept_rel_residual: float
    crack_length_threshold: float
    failure_threshold: float

    def run_id(self) -> str:
        return (
            f"reg{_safe_id(self.mech_regularization)}"
            f"_lim{_safe_id(self.mech_solution_abs_limit)}"
            f"_rel{_safe_id(self.mech_accept_rel_residual)}"
            f"_len{_safe_id(self.crack_length_threshold)}"
            f"_fail{_safe_id(self.failure_threshold)}"
        )


def _candidate_grid(args: argparse.Namespace) -> list[Candidate]:
    out: list[Candidate] = []
    for reg, limit, rel, len_th, fail_th in itertools.product(
        _parse_floats(args.mech_regularization_values),
        _parse_floats(args.mech_solution_abs_limit_values),
        _parse_floats(args.mech_accept_rel_residual_values),
        _parse_floats(args.crack_length_threshold_values),
        _parse_floats(args.failure_threshold_values),
    ):
        out.append(
            Candidate(
                mech_regularization=float(reg),
                mech_solution_abs_limit=float(limit),
                mech_accept_rel_residual=float(rel),
                crack_length_threshold=float(len_th),
                failure_threshold=float(fail_th),
            )
        )
    return out


def _run_scan(
    *,
    cfg_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    cmd = [
        sys.executable,
        str(ROOT / "sim/tests/scan_crack_onset.py"),
        "--config",
        str(cfg_path),
        "--out",
        str(out_dir),
    ]
    if not bool(args.scan_auto_output):
        cmd.append("--no-auto-output")
    if args.max_cases is not None:
        cmd.extend(["--max-cases", str(int(args.max_cases))])
    if args.min_onset_cases is not None:
        cmd.extend(["--min-onset-cases", str(int(args.min_onset_cases))])
    if args.max_runtime_warnings is not None:
        cmd.extend(["--max-runtime-warnings", str(int(args.max_runtime_warnings))])
    if args.max_mechanical_not_accepted_steps is not None:
        cmd.extend(
            [
                "--max-mechanical-not-accepted-steps",
                str(int(args.max_mechanical_not_accepted_steps)),
            ]
        )
    if args.max_crack_cg_nonconverged_steps is not None:
        cmd.extend(
            [
                "--max-crack-cg-nonconverged-steps",
                str(int(args.max_crack_cg_nonconverged_steps)),
            ]
        )
    if args.min_notch_cycles_completed is not None:
        cmd.extend(
            ["--min-notch-cycles-completed", str(int(args.min_notch_cycles_completed))]
        )
    timeout_s = None
    if args.scan_timeout_s is not None and float(args.scan_timeout_s) > 0:
        timeout_s = float(args.scan_timeout_s)
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
        return proc, False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        if stderr:
            stderr += "\n"
        stderr += f"[DOE] timeout after {timeout_s}s\n"
        proc = subprocess.CompletedProcess(cmd, 124, stdout, stderr)
        return proc, True


def _notch_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in cases if bool(c.get("notch_case", False))]


def _metric_notch_clip_steps(case_row: dict[str, Any]) -> int:
    diag = case_row.get("stability_diagnostics", {})
    if isinstance(diag, dict):
        return int(diag.get("mechanical_solution_clipped_steps", 0))
    return 0


def _metric_notch_crack_cg(case_row: dict[str, Any]) -> int:
    diag = case_row.get("stability_diagnostics", {})
    if isinstance(diag, dict):
        return int(diag.get("crack_cg_nonconverged_steps", 0))
    return 0


def _metric_notch_steps(case_row: dict[str, Any]) -> int:
    diag = case_row.get("stability_diagnostics", {})
    if isinstance(diag, dict):
        return int(diag.get("steps", 0))
    return 0


def _rank_key(run_row: dict[str, Any]) -> tuple[Any, ...]:
    # Better first: pass + length-led + low clipping + low CG noise.
    return (
        0 if bool(run_row.get("passed", False)) else 1,
        0 if bool(run_row.get("checks_passed", False)) else 1,
        0 if bool(run_row.get("notch_onset_length_all", False)) else 1,
        int(run_row.get("notch_not_accepted_max", 10**9)),
        int(run_row.get("notch_clip_total", 10**9)),
        int(run_row.get("notch_crack_cg_max", 10**9)),
        -int(run_row.get("notch_cycles_min", -1)),
        float(run_row.get("duration_s", 1e18)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DOE sweep on crack_onset_scan config.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("sim/configs/crack_onset_scan.yaml"),
        help="Base crack-onset YAML config.",
    )
    parser.add_argument("--tag", type=str, default="doe_week3_screen", help="Output tag.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root (default sim/tests/regress_runs/YYYY-MM-DD/<tag>).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on candidate count after Cartesian expansion.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=3,
        help="Cap scan cases (default 3 for notch-focused screening).",
    )
    parser.add_argument("--min-onset-cases", type=int, default=1)
    parser.add_argument("--max-runtime-warnings", type=int, default=50)
    parser.add_argument("--max-mechanical-not-accepted-steps", type=int, default=160)
    parser.add_argument("--max-crack-cg-nonconverged-steps", type=int, default=20)
    parser.add_argument("--min-notch-cycles-completed", type=int, default=3)
    parser.add_argument(
        "--scan-timeout-s",
        type=float,
        default=900.0,
        help="Timeout per scan run in seconds (<=0 disables timeout).",
    )
    parser.add_argument(
        "--scan-auto-output",
        action="store_true",
        default=False,
        help="Keep scan auto outputs (VTK/LAMMPS). Default off for DOE speed.",
    )
    parser.add_argument(
        "--vc-cycles",
        type=int,
        default=None,
        help="Override defaults.virtual_cycle.cycles for all DOE candidates.",
    )
    parser.add_argument(
        "--vc-cycle-points",
        type=int,
        default=None,
        help="Override defaults.virtual_cycle.cycle_points for all DOE candidates.",
    )
    parser.add_argument(
        "--vc-mech-max-iters",
        type=int,
        default=None,
        help="Override defaults.virtual_cycle.mech_max_iters for all DOE candidates.",
    )
    parser.add_argument(
        "--vc-mech-outer-max-iters",
        type=int,
        default=None,
        help="Override defaults.virtual_cycle.mech_outer_max_iters for all DOE candidates.",
    )
    parser.add_argument(
        "--vc-mech-tol",
        type=float,
        default=None,
        help="Override defaults.virtual_cycle.mech_tol for all DOE candidates.",
    )
    parser.add_argument(
        "--vc-mech-outer-tol",
        type=float,
        default=None,
        help="Override defaults.virtual_cycle.mech_outer_tol for all DOE candidates.",
    )
    parser.add_argument(
        "--vc-crack-tol",
        type=float,
        default=None,
        help="Override defaults.virtual_cycle.crack_tol for all DOE candidates.",
    )
    parser.add_argument(
        "--vc-crack-max-iters",
        type=int,
        default=None,
        help="Override defaults.virtual_cycle.crack_max_iters for all DOE candidates.",
    )
    parser.add_argument(
        "--vc-crack-accept-rel-residual",
        type=float,
        default=None,
        help="Override defaults.virtual_cycle.crack_accept_rel_residual for all DOE candidates.",
    )

    parser.add_argument("--mech-regularization-values", type=str, default="1.5,2.0,2.5")
    parser.add_argument("--mech-solution-abs-limit-values", type=str, default="8,10,12")
    parser.add_argument("--mech-accept-rel-residual-values", type=str, default="0.008,0.01,0.015")
    parser.add_argument("--crack-length-threshold-values", type=str, default="0.995")
    parser.add_argument("--failure-threshold-values", type=str, default="0.999")
    args = parser.parse_args()

    out_root = args.out or _default_out_dir(args.tag)
    out_root.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(args.base_config)
    candidates = _candidate_grid(args)
    if args.max_runs is not None:
        candidates = candidates[: max(0, int(args.max_runs))]
    if not candidates:
        raise ValueError("No candidates to run.")

    runs_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for idx, cand in enumerate(candidates, start=1):
        run_name = f"run_{idx:03d}_{cand.run_id()}"
        run_dir = out_root / run_name
        scan_out = run_dir / "scan"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(candidates)}] {run_name} ...", flush=True)

        cfg = json.loads(json.dumps(base_cfg))
        vc, criteria = _ensure_defaults(cfg)
        vc["mech_regularization"] = cand.mech_regularization
        vc["mech_solution_abs_limit"] = cand.mech_solution_abs_limit
        vc["mech_accept_rel_residual"] = cand.mech_accept_rel_residual
        vc["crack_length_threshold"] = cand.crack_length_threshold
        vc["failure_threshold"] = cand.failure_threshold
        if args.vc_cycles is not None:
            vc["cycles"] = int(args.vc_cycles)
        if args.vc_cycle_points is not None:
            vc["cycle_points"] = int(args.vc_cycle_points)
        if args.vc_mech_max_iters is not None:
            vc["mech_max_iters"] = int(args.vc_mech_max_iters)
        if args.vc_mech_outer_max_iters is not None:
            vc["mech_outer_max_iters"] = int(args.vc_mech_outer_max_iters)
        if args.vc_mech_tol is not None:
            vc["mech_tol"] = float(args.vc_mech_tol)
        if args.vc_mech_outer_tol is not None:
            vc["mech_outer_tol"] = float(args.vc_mech_outer_tol)
        if args.vc_crack_tol is not None:
            vc["crack_tol"] = float(args.vc_crack_tol)
        if args.vc_crack_max_iters is not None:
            vc["crack_max_iters"] = int(args.vc_crack_max_iters)
        if args.vc_crack_accept_rel_residual is not None:
            vc["crack_accept_rel_residual"] = float(args.vc_crack_accept_rel_residual)
        criteria["min_onset_cases"] = int(args.min_onset_cases)
        criteria["max_runtime_warnings"] = int(args.max_runtime_warnings)
        criteria["max_mechanical_not_accepted_steps"] = int(args.max_mechanical_not_accepted_steps)
        criteria["max_crack_cg_nonconverged_steps"] = int(args.max_crack_cg_nonconverged_steps)
        criteria["min_notch_cycles_completed"] = int(args.min_notch_cycles_completed)

        cfg_path = run_dir / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        proc, timed_out = _run_scan(cfg_path=cfg_path, out_dir=scan_out, args=args)
        (run_dir / "scan.stdout").write_text(proc.stdout, encoding="utf-8")
        (run_dir / "scan.stderr").write_text(proc.stderr, encoding="utf-8")

        summary_path = scan_out / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = {
                "passed": False,
                "duration_s": None,
                "onset_cases": 0,
                "checks_passed": False,
                "failure_reasons": [f"scan_summary_missing(exit={proc.returncode})"],
                "cases": [],
            }

        cases = list(summary.get("cases", []))
        notch_cases = _notch_rows(cases)
        notch_clip_steps = [_metric_notch_clip_steps(c) for c in notch_cases]
        notch_steps = [_metric_notch_steps(c) for c in notch_cases]
        notch_not_accepted = [int(c.get("mech_not_accepted_steps", 0)) for c in notch_cases]
        notch_crack_cg = [_metric_notch_crack_cg(c) for c in notch_cases]
        notch_cycles = [int(c.get("cycles_completed", 0)) for c in notch_cases]
        notch_onset_length_all = all(bool(c.get("onset_length", False)) for c in notch_cases) if notch_cases else False

        run_row = {
            "run_name": run_name,
            "config_path": str(cfg_path),
            "summary_path": str(summary_path),
            "scan_exit_code": int(proc.returncode),
            "timed_out": bool(timed_out),
            "passed": bool(summary.get("passed", False)),
            "checks_passed": bool(summary.get("checks_passed", False)),
            "onset_cases": int(summary.get("onset_cases", 0)),
            "duration_s": float(summary.get("duration_s", 0.0) or 0.0),
            "failure_reasons": ";".join(str(x) for x in summary.get("failure_reasons", [])),
            "notch_cases": len(notch_cases),
            "notch_onset_length_all": notch_onset_length_all,
            "notch_clip_total": int(sum(notch_clip_steps)),
            "notch_clip_max": int(max(notch_clip_steps) if notch_clip_steps else 0),
            "notch_steps_total": int(sum(notch_steps)),
            "notch_clip_ratio": (
                float(sum(notch_clip_steps)) / float(sum(notch_steps)) if notch_steps and sum(notch_steps) > 0 else 0.0
            ),
            "notch_not_accepted_max": int(max(notch_not_accepted) if notch_not_accepted else 0),
            "notch_crack_cg_max": int(max(notch_crack_cg) if notch_crack_cg else 0),
            "notch_cycles_min": int(min(notch_cycles) if notch_cycles else 0),
            "mech_regularization": cand.mech_regularization,
            "mech_solution_abs_limit": cand.mech_solution_abs_limit,
            "mech_accept_rel_residual": cand.mech_accept_rel_residual,
            "crack_length_threshold": cand.crack_length_threshold,
            "failure_threshold": cand.failure_threshold,
        }
        runs_rows.append(run_row)
        print(
            f"[{idx}/{len(candidates)}] done exit={proc.returncode} "
            f"passed={run_row['passed']} notch_clip_total={run_row['notch_clip_total']} "
            f"notch_not_acc_max={run_row['notch_not_accepted_max']}",
            flush=True,
        )

        for c in cases:
            case_rows.append(
                {
                    "run_name": run_name,
                    "case_name": str(c.get("name", "")),
                    "passed": bool(c.get("passed", False)),
                    "checks_ok": bool(c.get("checks_ok", False)),
                    "onset": bool(c.get("onset", False)),
                    "onset_length": bool(c.get("onset_length", False)),
                    "onset_mean_aux": bool(c.get("onset_mean_aux", False)),
                    "notch_case": bool(c.get("notch_case", False)),
                    "cycles_completed": int(c.get("cycles_completed", 0)),
                    "crack_mean_final": float(c.get("crack_mean_final", 0.0)),
                    "crack_mean_delta": float(c.get("crack_mean_delta", 0.0)),
                    "mech_not_accepted_steps": int(c.get("mech_not_accepted_steps", 0)),
                    "mechanical_solution_clipped_steps": int(_metric_notch_clip_steps(c)),
                    "steps": int(_metric_notch_steps(c)),
                    "crack_cg_nonconverged_steps": int(_metric_notch_crack_cg(c)),
                }
            )

    runs_sorted = sorted(runs_rows, key=_rank_key)
    for rank, row in enumerate(runs_sorted, start=1):
        row["rank"] = rank

    runs_csv = out_root / "runs.csv"
    if runs_sorted:
        with runs_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(runs_sorted[0].keys()))
            writer.writeheader()
            writer.writerows(runs_sorted)

    cases_csv = out_root / "cases.csv"
    if case_rows:
        with cases_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(case_rows[0].keys()))
            writer.writeheader()
            writer.writerows(case_rows)

    summary = {
        "base_config": str(args.base_config),
        "out_root": str(out_root),
        "candidate_count": len(candidates),
        "completed_runs": len(runs_rows),
        "wall_time_s": time.perf_counter() - t0,
        "top_runs": runs_sorted[:5],
        "runs_csv": str(runs_csv),
        "cases_csv": str(cases_csv),
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"summary_path": str(summary_path), "runs_csv": str(runs_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
