"""
Plot baseline GND cycle statistics (gnd_mean vs accum_plastic_mean).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GND cycle baseline from CSV.")
    parser.add_argument("--input", type=Path, required=True, help="gnd_cycle.csv path")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--title", type=str, default="GND vs Accumulated Plastic (Cycle)")
    args = parser.parse_args()

    data = np.genfromtxt(args.input, delimiter=",", names=True)
    cycles = data["cycle"]
    gnd_mean = data["gnd_mean"]
    accum = data["accum_plastic_mean"]

    output = args.output
    if output is None:
        output = args.input.with_name("gnd_cycle_baseline.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax1.plot(cycles, gnd_mean, "o-", color="#1f77b4", label="gnd_mean")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("GND mean", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(cycles, accum, "s--", color="#d62728", label="accum_plastic_mean")
    ax2.set_ylabel("Accumulated plastic mean", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
