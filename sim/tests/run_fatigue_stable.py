"""
Run cyclic simulation until stable response and write outputs to a run directory.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from sim.tests.virtual_cycle import run_virtual_cycles
except ImportError:  # pragma: no cover - allow running in-place
    from virtual_cycle import run_virtual_cycles  # type: ignore


def _default_run_dir(tag: str) -> Path:
    date_str = date.today().isoformat()
    time_str = datetime.now().strftime("%H%M%S")
    return Path("sim/tests/runs") / date_str / f"{tag}_{time_str}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fatigue cycles to stable response.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--tag", type=str, default="fatigue_exp_match_200pts", help="Run directory tag.")
    parser.add_argument("--cycles", type=int, default=200, help="Max cycles.")
    parser.add_argument("--max-strain", type=float, default=0.001286103, help="Max macro strain.")
    parser.add_argument("--cycle-points", type=int, default=200, help="Points per cycle.")
    parser.add_argument("--orientation", type=str, default="-1,1,1", help="Orientation vector, e.g. -1,1,1")
    parser.add_argument("--stable-window", type=int, default=5, help="Stable window (cycles).")
    parser.add_argument("--stable-tol", type=float, default=0.02, help="Stable tolerance (relative).")
    parser.add_argument("--toughness-scale", type=float, default=0.1, help="Toughness scale.")
    parser.add_argument("--yield-tau", type=float, default=None, help="Override yield_tau (nd).")
    parser.add_argument("--flow-scale", type=float, default=None, help="Override flow_scale (nd).")
    parser.add_argument("--linear-hardening", type=float, default=None, help="Override linear_hardening (nd).")
    parser.add_argument("--visco-exponent", type=float, default=None, help="Override visco_exponent.")
    parser.add_argument("--gamma0", type=float, default=None, help="Override slip gamma0.")
    parser.add_argument("--slip-exponent", type=float, default=None, help="Override slip exponent n.")
    parser.add_argument("--h-iso", type=float, default=None, help="Override isotropic hardening h_iso.")
    parser.add_argument("--h-gnd", type=float, default=None, help="Override GND hardening h_gnd.")
    parser.add_argument("--mech-max-iters", type=int, default=None, help="Mechanical CG max iters.")
    parser.add_argument("--mech-tol", type=float, default=None, help="Mechanical CG tolerance.")
    parser.add_argument("--mech-outer-max-iters", type=int, default=None, help="Unilateral outer iters.")
    parser.add_argument("--mech-outer-tol", type=float, default=None, help="Unilateral outer tol.")
    parser.add_argument("--pfc-active", action="store_true", help="Enable PFC evolution.")
    parser.add_argument("--pfc-frozen", action="store_true", help="Disable PFC evolution.")
    parser.add_argument("--gnd-active", action="store_true", help="Enable GND diagnostics.")
    parser.add_argument("--gnd-burgers", type=float, default=1.0, help="Burgers vector scale (nd).")
    args = parser.parse_args()

    orientation = tuple(float(v.strip()) for v in args.orientation.split(","))
    if len(orientation) != 3:
        raise ValueError("orientation must have 3 components, e.g. -1,1,1")

    run_dir = args.run_dir or _default_run_dir(args.tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    pfc_active = True
    if args.pfc_frozen:
        pfc_active = False
    elif args.pfc_active:
        pfc_active = True

    results, paris, coffman = run_virtual_cycles(
        cycles=args.cycles,
        max_strain=args.max_strain,
        cycle_points=args.cycle_points,
        orientation_vector=orientation,
        stable_window=args.stable_window,
        stable_tol=args.stable_tol,
        stable_metrics=("plastic_range", "rss_peak_nd"),
        defect_config=None,
        pre_relax_steps=0,
        pre_relax_strain=0.0,
        toughness_scale=args.toughness_scale,
        yield_tau=args.yield_tau,
        flow_scale=args.flow_scale,
        linear_hardening=args.linear_hardening,
        visco_exponent=args.visco_exponent,
        gamma0=args.gamma0,
        slip_exponent=args.slip_exponent,
        h_iso=args.h_iso,
        h_gnd=args.h_gnd,
        mech_max_iters=args.mech_max_iters,
        mech_tol=args.mech_tol,
        mech_outer_max_iters=args.mech_outer_max_iters,
        mech_outer_tol=args.mech_outer_tol,
        pfc_active=pfc_active,
        gnd_active=args.gnd_active,
        gnd_burgers=args.gnd_burgers,
        csv_output=run_dir / "virtual_cycle.csv",
        analysis_csv=run_dir / "virtual_cycle_analysis.csv",
        stress_strain_csv=run_dir / "virtual_cycle_stress_strain.csv",
        data_output=None,
        vtk_dir=None,
        dump_dir=None,
        initial_vtk=None,
    )

    print(f"run_dir={run_dir}")
    print(f"cycles_run={len(results)} paris={paris:.6e} coffman={coffman:.6e}")
    if results:
        last = results[-1]
        print(
            f"crack_mean={last.crack_mean:.6e} crack_len={last.crack_length:.6e} "
            f"accum_plastic_mean={last.accum_plastic_mean:.6e} plastic_range={last.plastic_range:.6e} "
            f"rss_peak_nd={last.rss_peak_nd:.6e}"
        )


if __name__ == "__main__":
    main()
