"""
Utility to generate a standalone 111-oriented defected single-crystal copper sample.
"""

from __future__ import annotations

from pathlib import Path

from ..operators import GridSpec
from ..structure import Cu111StructureBuilder


def main() -> None:
    grid = GridSpec(shape=(32, 32, 16), spacing=(50, 50, 50), periodic=(True, True, False))
    builder = Cu111StructureBuilder(grid, defect_fraction=0.08, defect_amplitude=0.3)
    structure = builder.build(seed=123)
    data_path = Path("sim/tests/cu111_single.data")
    dump_path = Path("sim/tests/cu111_single.lammpstrj")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    structure.export(data_path, dump_path)
    print(f"Generated {data_path} and {dump_path} for visualization.")


if __name__ == "__main__":
    main()
