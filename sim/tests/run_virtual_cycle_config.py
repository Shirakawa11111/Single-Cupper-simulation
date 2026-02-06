"""
Run `run_virtual_cycles` from a YAML config file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # type: ignore

from sim.tests.virtual_cycle import run_virtual_cycles

PATH_KEYS = {
    "csv_output",
    "analysis_csv",
    "data_output",
    "dump_dir",
    "vtk_dir",
    "initial_vtk",
    "stress_strain_csv",
    "run_dir",
}
TUPLE_KEYS = {
    "orientation_vector",
    "grid_shape",
    "grid_spacing",
    "grid_periodic",
    "stable_metrics",
}


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    return value


def _normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    for key in PATH_KEYS:
        if key in out and out[key] is not None:
            out[key] = Path(out[key])
    for key in TUPLE_KEYS:
        if key in out and isinstance(out[key], list):
            out[key] = tuple(out[key])
    if "notch_box" in out and isinstance(out["notch_box"], list):
        out["notch_box"] = tuple(tuple(row) for row in out["notch_box"])
    if "grid_shape" in out and out["grid_shape"] is not None:
        out["grid_shape"] = tuple(int(v) for v in out["grid_shape"])
    if "grid_periodic" in out and out["grid_periodic"] is not None:
        out["grid_periodic"] = tuple(bool(v) for v in out["grid_periodic"])
    if "grid_spacing" in out and out["grid_spacing"] is not None:
        out["grid_spacing"] = tuple(float(v) for v in out["grid_spacing"])
    if "orientation_vector" in out and out["orientation_vector"] is not None:
        out["orientation_vector"] = tuple(float(v) for v in out["orientation_vector"])
    return out


def _resolve_payload(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "virtual_cycle" in raw:
        vc = raw.get("virtual_cycle")
        if not isinstance(vc, dict):
            raise ValueError("virtual_cycle must be a mapping.")
        meta = {k: v for k, v in raw.items() if k != "virtual_cycle"}
        return vc, meta
    return raw, {}


def _resolve_summary_dir(cfg: dict[str, Any]) -> Path | None:
    for key in ("run_dir", "csv_output", "analysis_csv", "stress_strain_csv"):
        val = cfg.get(key)
        if val is None:
            continue
        p = Path(val)
        return p if key == "run_dir" else p.parent
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run virtual_cycle from YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional summary JSON output path.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved kwargs and exit.")
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")

    vc_cfg_raw, meta = _resolve_payload(raw)
    cfg = _normalize_config(vc_cfg_raw)

    if args.dry_run:
        print(json.dumps({"config": _to_builtin(cfg), "meta": _to_builtin(meta)}, indent=2))
        return 0

    t0 = datetime.now()
    results, paris_coeff, coffman = run_virtual_cycles(**cfg)
    t1 = datetime.now()

    last = results[-1] if results else None
    summary = {
        "config_path": str(args.config),
        "started_at": t0.isoformat(timespec="seconds"),
        "finished_at": t1.isoformat(timespec="seconds"),
        "duration_s": (t1 - t0).total_seconds(),
        "meta": _to_builtin(meta),
        "config": _to_builtin(cfg),
        "cycles_completed": len(results),
        "paris_coeff": float(paris_coeff),
        "coffman_coeff": float(coffman),
        "last_cycle": None if last is None else _to_builtin(last.__dict__),
    }

    out_dir = _resolve_summary_dir(cfg)
    summary_path = args.summary_output
    if summary_path is None:
        if out_dir is not None:
            summary_path = out_dir / "run_summary.json"
        else:
            day = date.today().isoformat()
            ts = datetime.now().strftime("%H%M%S")
            summary_path = Path("sim/tests/runs") / day / f"config_run_{args.config.stem}_{ts}" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
