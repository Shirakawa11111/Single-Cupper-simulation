"""
Compare GND cycle trends for two runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load(csv_path: Path) -> dict:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return {
        "cycle": data["cycle"],
        "gnd_mean": data["gnd_mean"],
        "accum_plastic_mean": data["accum_plastic_mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GND cycle trends between two runs.")
    parser.add_argument("--a", type=Path, required=True, help="First gnd_cycle.csv")
    parser.add_argument("--b", type=Path, required=True, help="Second gnd_cycle.csv")
    parser.add_argument("--label-a", type=str, default="h_gnd=0")
    parser.add_argument("--label-b", type=str, default="h_gnd=1e-4")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    data_a = _load(args.a)
    data_b = _load(args.b)

    output = args.output
    if output is None:
        output = args.a.parent / "gnd_cycle_compare.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True)
    ax1.plot(data_a["cycle"], data_a["gnd_mean"], "o-", label=f"{args.label_a}", color="#1f77b4")
    ax1.plot(data_b["cycle"], data_b["gnd_mean"], "s--", label=f"{args.label_b}", color="#ff7f0e")
    ax1.set_ylabel("GND mean")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.plot(data_a["cycle"], data_a["accum_plastic_mean"], "o-", label=f"{args.label_a}", color="#1f77b4")
    ax2.plot(data_b["cycle"], data_b["accum_plastic_mean"], "s--", label=f"{args.label_b}", color="#ff7f0e")
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Accumulated plastic mean")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Low-amplitude cycle: GND vs Accumulated Plastic")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
