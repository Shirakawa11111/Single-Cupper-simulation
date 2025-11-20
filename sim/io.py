"""
Shared I/O helpers for exporting fields to LAMMPS-friendly formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .operators import GridSpec


def write_atomic_data(path: Path, grid: GridSpec, mass: float = 63.546) -> None:
    nx, ny, nz = grid.shape
    dx, dy, dz = grid.spacing
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz
    coords_x = np.linspace(0, Lx - dx, nx)
    coords_y = np.linspace(0, Ly - dy, ny)
    coords_z = np.linspace(0, Lz - dz, nz)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("LAMMPS data file generated for OVITO\n\n")
        total = nx * ny * nz
        fh.write(f"{total} atoms\n")
        fh.write("1 atom types\n\n")
        fh.write(f"0.0 {Lx:.6e} xlo xhi\n")
        fh.write(f"0.0 {Ly:.6e} ylo yhi\n")
        fh.write(f"0.0 {Lz:.6e} zlo zhi\n\n")
        fh.write(f"Masses\n\n1 {mass:.6f}\n\n")
        fh.write("Atoms # atomic\n\n")
        atom_id = 1
        for k, z in enumerate(coords_z):
            for j, y in enumerate(coords_y):
                for i, x in enumerate(coords_x):
                    fh.write(f"{atom_id} 1 {x:.6e} {y:.6e} {z:.6e} 0 0 0\n")
                    atom_id += 1


def write_lammpstrj(path: Path, grid: GridSpec, fields: Dict[str, np.ndarray], timestep: int = 0) -> None:
    nx, ny, nz = grid.shape
    dx, dy, dz = grid.spacing
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz
    coords_x = np.linspace(0, Lx - dx, nx)
    coords_y = np.linspace(0, Ly - dy, ny)
    coords_z = np.linspace(0, Lz - dz, nz)
    crack = fields.get("crack")
    plastic = fields.get("plastic")
    psi = fields.get("psi")
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ITEM: TIMESTEP\n")
        fh.write(f"{timestep}\n")
        fh.write("ITEM: NUMBER OF ATOMS\n")
        fh.write(f"{nx * ny * nz}\n")
        fh.write("ITEM: BOX BOUNDS pp pp pp\n")
        fh.write(f"0.0 {Lx:.6e}\n")
        fh.write(f"0.0 {Ly:.6e}\n")
        fh.write(f"0.0 {Lz:.6e}\n")
        fh.write("ITEM: ATOMS id type x y z crack plastic psi\n")
        atom_id = 1
        for k, z in enumerate(coords_z):
            for j, y in enumerate(coords_y):
                for i, x in enumerate(coords_x):
                    fh.write(
                        f"{atom_id} 1 {x:.6e} {y:.6e} {z:.6e} "
                        f"{crack[i, j, k]:.6e} {plastic[i, j, k]:.6e} {psi[i, j, k]:.6e}\n"
                    )
                    atom_id += 1
