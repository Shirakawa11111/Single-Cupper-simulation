"""
Scan GND sensitivity to crystal orientation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.tests.virtual_cycle import run_virtual_cycles


def _parse_orientations(text: str) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for chunk in text.split(";"):
        vals = [v.strip() for v in chunk.split(",") if v.strip()]
        if len(vals) != 3:
            raise ValueError(f"Invalid orientation: {chunk}")
        out.append(tuple(float(v) for v in vals))
    return out


def _default_orientations() -> List[Tuple[float, float, float]]:
    return [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 2.0)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan GND statistics for multiple orientations.")
    parser.add_argument("--output-root", type=Path, default=None, help="Root output dir.")
    parser.add_argument("--orientations", type=str, default="", help="Semicolon-separated list, e.g. 1,0,0;1,1,0")
    parser.add_argument("--cycles", type=int, default=3, help="Cycles per orientation.")
    parser.add_argument("--max-strain", type=float, default=0.005, help="Max strain amplitude.")
    parser.add_argument("--cycle-points", type=int, default=80, help="Points per cycle.")
    parser.add_argument("--pfc-active", action="store_true", help="Enable PFC evolution.")
    parser.add_argument("--tag", type=str, default="gnd_orient_scan", help="Tag for output directory.")
    args = parser.parse_args()

    orientations = _parse_orientations(args.orientations) if args.orientations else _default_orientations()
    date_str = date.today().isoformat()
    time_str = datetime.now().strftime("%H%M%S")
    root = args.output_root or Path("sim/tests/runs") / date_str / f"{args.tag}_{time_str}"
    root.mkdir(parents=True, exist_ok=True)

    params = {
        "script": "scan_gnd_orientations",
        "args": {
            "orientations": orientations,
            "cycles": args.cycles,
            "max_strain": args.max_strain,
            "cycle_points": args.cycle_points,
            "pfc_active": args.pfc_active,
        },
    }
    (root / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    for ori in orientations:
        ori_tag = f"ori_{ori[0]:g}_{ori[1]:g}_{ori[2]:g}".replace(".", "p")
        run_dir = root / ori_tag
        run_dir.mkdir(parents=True, exist_ok=True)
        results, _, _ = run_virtual_cycles(
            cycles=args.cycles,
            max_strain=args.max_strain,
            cycle_points=args.cycle_points,
            orientation_vector=ori,
            gnd_active=True,
            pfc_active=args.pfc_active,
            stable_window=None,
            csv_output=run_dir / "virtual_cycle.csv",
            analysis_csv=run_dir / "virtual_cycle_analysis.csv",
            stress_strain_csv=run_dir / "virtual_cycle_stress_strain.csv",
            vtk_dir=run_dir / "vtk",
            dump_dir=None,
            initial_vtk=None,
        )
        summary = {
            "orientation": ori,
            "cycles": len(results),
            "gnd_mean_last": results[-1].gnd_mean if results else 0.0,
            "gnd_max_last": results[-1].gnd_max if results else 0.0,
            "accum_plastic_last": results[-1].accum_plastic_mean if results else 0.0,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
