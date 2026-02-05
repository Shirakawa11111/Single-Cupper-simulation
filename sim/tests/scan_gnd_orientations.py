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
    parser.add_argument("--grid-shape", type=str, default="32,16,8", help="Grid shape Nx,Ny,Nz.")
    parser.add_argument("--grid-spacing", type=str, default="1,1,1", help="Grid spacing dx,dy,dz.")
    parser.add_argument("--h-gnd-list", type=str, default="", help="Comma list of h_gnd values.")
    parser.add_argument("--pfc-active", action="store_true", help="Enable PFC evolution.")
    parser.add_argument("--notch", action="store_true", help="Seed a default notch to induce gradients.")
    parser.add_argument("--notch-box", type=str, default="", help="Custom notch box x0,x1,y0,y1,z0,z1")
    parser.add_argument("--tag", type=str, default="gnd_orient_scan", help="Tag for output directory.")
    args = parser.parse_args()

    orientations = _parse_orientations(args.orientations) if args.orientations else _default_orientations()
    date_str = date.today().isoformat()
    time_str = datetime.now().strftime("%H%M%S")
    root = args.output_root or Path("sim/tests/runs") / date_str / f"{args.tag}_{time_str}"
    root.mkdir(parents=True, exist_ok=True)

    shape_vals = [int(v.strip()) for v in args.grid_shape.split(",") if v.strip()]
    if len(shape_vals) != 3:
        raise ValueError("grid_shape must have 3 comma-separated ints")
    spacing_vals = [float(v.strip()) for v in args.grid_spacing.split(",") if v.strip()]
    if len(spacing_vals) != 3:
        raise ValueError("grid_spacing must have 3 comma-separated floats")
    grid_shape = tuple(shape_vals)
    grid_spacing = tuple(spacing_vals)

    notch_box = None
    if args.notch_box:
        vals = [float(v.strip()) for v in args.notch_box.split(",") if v.strip()]
        if len(vals) != 6:
            raise ValueError("notch_box must have 6 comma-separated values")
        notch_box = ((vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5]))
    elif args.notch:
        nx, ny, nz = grid_shape
        dx, dy, dz = grid_spacing
        notch_box = (
            (0.35 * nx * dx, 0.45 * nx * dx),
            (0.45 * ny * dy, 0.55 * ny * dy),
            (0.35 * nz * dz, 0.65 * nz * dz),
        )

    h_gnd_list = [float(v.strip()) for v in args.h_gnd_list.split(",") if v.strip()] or [0.0]

    params = {
        "script": "scan_gnd_orientations",
        "args": {
            "orientations": orientations,
            "cycles": args.cycles,
            "max_strain": args.max_strain,
            "cycle_points": args.cycle_points,
            "grid_shape": grid_shape,
            "grid_spacing": grid_spacing,
            "pfc_active": args.pfc_active,
            "notch_box": notch_box,
            "h_gnd_list": h_gnd_list,
        },
    }
    (root / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    summary_rows = []
    for ori in orientations:
        ori_tag = f"ori_{ori[0]:g}_{ori[1]:g}_{ori[2]:g}".replace(".", "p")
        ori_dir = root / ori_tag
        ori_dir.mkdir(parents=True, exist_ok=True)
        for h_gnd in h_gnd_list:
            h_tag = f"hgnd_{h_gnd:.0e}".replace("+", "").replace("-", "m").replace(".", "p")
            run_dir = ori_dir / h_tag
            run_dir.mkdir(parents=True, exist_ok=True)
            results, _, _ = run_virtual_cycles(
                cycles=args.cycles,
                max_strain=args.max_strain,
                cycle_points=args.cycle_points,
                orientation_vector=ori,
                grid_shape=grid_shape,
                grid_spacing=grid_spacing,
                gnd_active=True,
                pfc_active=args.pfc_active,
                h_gnd=h_gnd,
                notch_box=notch_box,
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
                "h_gnd": h_gnd,
                "cycles": len(results),
                "gnd_mean_last": results[-1].gnd_mean if results else 0.0,
                "gnd_max_last": results[-1].gnd_max if results else 0.0,
                "accum_plastic_last": results[-1].accum_plastic_mean if results else 0.0,
            }
            summary_rows.append(summary)
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_path = root / "summary.csv"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write("orientation,h_gnd,gnd_mean_last,gnd_max_last,accum_plastic_last\n")
        for row in summary_rows:
            o = row["orientation"]
            fh.write(
                f"{o[0]} {o[1]} {o[2]},{row['h_gnd']},{row['gnd_mean_last']:.6e},"
                f"{row['gnd_max_last']:.6e},{row['accum_plastic_last']:.6e}\n"
            )

    # Plot: gnd_mean_last vs h_gnd for each orientation
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
        import numpy as np  # noqa: E402

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for ori in orientations:
            vals = [r for r in summary_rows if r["orientation"] == ori]
            vals = sorted(vals, key=lambda x: x["h_gnd"])
            hs = [v["h_gnd"] for v in vals]
            gm = [v["gnd_mean_last"] for v in vals]
            label = f"[{ori[0]:g}{ori[1]:g}{ori[2]:g}]"
            ax.plot(hs, gm, marker="o", label=label)
        ax.set_xlabel("h_gnd")
        ax.set_ylabel("GND mean (last cycle)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(root / "gnd_mean_vs_hgnd.png", dpi=160)

        fig2, ax2 = plt.subplots(figsize=(6.4, 4.2))
        for ori in orientations:
            vals = [r for r in summary_rows if r["orientation"] == ori]
            vals = sorted(vals, key=lambda x: x["h_gnd"])
            hs = [v["h_gnd"] for v in vals]
            am = [v["accum_plastic_last"] for v in vals]
            label = f"[{ori[0]:g}{ori[1]:g}{ori[2]:g}]"
            ax2.plot(hs, am, marker="s", label=label)
        ax2.set_xlabel("h_gnd")
        ax2.set_ylabel("Accumulated plastic (last cycle)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left")
        fig2.tight_layout()
        fig2.savefig(root / "accum_plastic_vs_hgnd.png", dpi=160)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] plot failed: {exc}")


if __name__ == "__main__":
    main()
