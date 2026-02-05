"""
Scan h_gnd values for low-amplitude cyclic response and auto-plot.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.tests.regress_gnd_cycle import _run_cycles  # type: ignore


def _parse_hgnd_list(text: str) -> List[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _tag(val: float) -> str:
    text = f"{val:.0e}" if val != 0 else "0"
    return text.replace("+", "").replace("-", "m")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan h_gnd values for low-amplitude GND cycles.")
    parser.add_argument("--output-root", type=Path, default=None, help="Root output dir.")
    parser.add_argument("--h-gnd-list", type=str, default="0,5e-5,1e-4,2e-4", help="Comma list of h_gnd.")
    parser.add_argument("--cycles", type=int, default=5, help="Number of cycles.")
    parser.add_argument("--max-strain", type=float, default=5e-3, help="Max macro strain amplitude.")
    parser.add_argument("--segment-steps", type=int, default=20, help="Steps per segment.")
    parser.add_argument("--dt", type=float, default=5e-3, help="Time step.")
    parser.add_argument("--gamma0", type=float, default=1e-2, help="Slip rate scale.")
    parser.add_argument("--slip-exponent", type=float, default=8.0, help="Slip exponent n.")
    parser.add_argument("--h-iso", type=float, default=4e-4, help="Isotropic hardening.")
    parser.add_argument("--plastic-relax", type=float, default=0.2, help="Plastic relax factor.")
    parser.add_argument("--pfc-active", action="store_true", help="Enable PFC evolution.")
    parser.add_argument("--tag", type=str, default="gnd_cycle_hgnd_scan", help="Tag for output directory.")
    args = parser.parse_args()

    h_gnd_list = _parse_hgnd_list(args.h_gnd_list)
    date_str = date.today().isoformat()
    time_str = datetime.now().strftime("%H%M%S")
    root = args.output_root or Path("sim/tests/regress_runs") / date_str / f"{args.tag}_{time_str}"
    root.mkdir(parents=True, exist_ok=True)

    params = {
        "script": "scan_hgnd_cycle",
        "args": {
            "h_gnd_list": h_gnd_list,
            "cycles": args.cycles,
            "max_strain": args.max_strain,
            "segment_steps": args.segment_steps,
            "dt": args.dt,
            "gamma0": args.gamma0,
            "slip_exponent": args.slip_exponent,
            "h_iso": args.h_iso,
            "plastic_relax": args.plastic_relax,
            "pfc_active": args.pfc_active,
        },
    }
    (root / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    summary_rows = []
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True)

    for h_gnd in h_gnd_list:
        gnd_means, gnd_max, accum_means = _run_cycles(
            cycles=args.cycles,
            max_strain=args.max_strain,
            segment_steps=args.segment_steps,
            dt=args.dt,
            gamma0=args.gamma0,
            slip_exponent=args.slip_exponent,
            h_iso=args.h_iso,
            h_gnd=h_gnd,
            plastic_relax=args.plastic_relax,
            pfc_active=args.pfc_active,
        )
        cycles_arr = np.arange(1, len(gnd_means) + 1, dtype=int)
        out_dir = root / f"hgnd_{_tag(h_gnd)}"
        out_dir.mkdir(parents=True, exist_ok=True)

        gnd_growth = gnd_means[-1] - gnd_means[0]
        accum_growth = accum_means[-1] - accum_means[0]
        summary = {
            "h_gnd": h_gnd,
            "cycles": len(gnd_means),
            "gnd_mean_last": gnd_means[-1],
            "gnd_max_last": gnd_max[-1],
            "accum_plastic_last": accum_means[-1],
            "gnd_growth": gnd_growth,
            "accum_growth": accum_growth,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (out_dir / "gnd_cycle.csv").open("w", encoding="utf-8") as fh:
            fh.write("cycle,gnd_mean,gnd_max,accum_plastic_mean\n")
            for c, gm, gx, am in zip(cycles_arr, gnd_means, gnd_max, accum_means):
                fh.write(f"{c},{gm:.6e},{gx:.6e},{am:.6e}\n")

        summary_rows.append(summary)
        label = f"h_gnd={h_gnd:g}"
        ax1.plot(cycles_arr, gnd_means, marker="o", label=label)
        ax2.plot(cycles_arr, accum_means, marker="s", label=label)

    with (root / "summary.csv").open("w", encoding="utf-8") as fh:
        fh.write("h_gnd,gnd_mean_last,gnd_max_last,accum_plastic_last,gnd_growth,accum_growth\n")
        for row in summary_rows:
            fh.write(
                f"{row['h_gnd']},{row['gnd_mean_last']:.6e},{row['gnd_max_last']:.6e},"
                f"{row['accum_plastic_last']:.6e},{row['gnd_growth']:.6e},{row['accum_growth']:.6e}\n"
            )

    ax1.set_ylabel("GND mean")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Accumulated plastic mean")
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Low-amplitude cycle: h_gnd scan")
    fig.tight_layout()
    fig.savefig(root / "gnd_cycle_hgnd_scan.png", dpi=160)

    print(f"saved: {root / 'gnd_cycle_hgnd_scan.png'}")


if __name__ == "__main__":
    main()
