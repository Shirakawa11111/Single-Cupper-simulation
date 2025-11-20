"""
Shared I/O helpers for exporting fields to LAMMPS-friendly and VTK formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from .operators import GridSpec


def write_atomic_data(path: Path, grid: GridSpec, mass: float = 63.546) -> None:
    """
    Export grid points as atoms in LAMMPS data format (for OVITO).
    """
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
        for k in range(len(coords_z)):
            z = coords_z[k]
            for j in range(len(coords_y)):
                y = coords_y[j]
                for i in range(len(coords_x)):
                    x = coords_x[i]
                    fh.write(f"{atom_id} 1 {x:.6e} {y:.6e} {z:.6e} 0 0 0\n")
                    atom_id += 1


def write_lammpstrj(path: Path, grid: GridSpec, fields: Dict[str, np.ndarray], timestep: int = 0) -> None:
    """
    Export fields to LAMMPS trajectory format.
    """
    nx, ny, nz = grid.shape
    dx, dy, dz = grid.spacing
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz
    coords_x = np.linspace(0, Lx - dx, nx)
    coords_y = np.linspace(0, Ly - dy, ny)
    coords_z = np.linspace(0, Lz - dz, nz)
    
    crack = fields.get("crack", np.zeros((nx, ny, nz)))
    plastic = fields.get("plastic", np.zeros((nx, ny, nz)))
    psi = fields.get("psi", np.zeros((nx, ny, nz)))
    
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
        for k in range(nz):
            z = coords_z[k]
            for j in range(ny):
                y = coords_y[j]
                for i in range(nx):
                    x = coords_x[i]
                    fh.write(
                        f"{atom_id} 1 {x:.6e} {y:.6e} {z:.6e} "
                        f"{crack[i, j, k]:.6e} {plastic[i, j, k]:.6e} {psi[i, j, k]:.6e}\n"
                    )
                    atom_id += 1


def write_vtk(path: Path, grid: GridSpec, fields: Dict[str, np.ndarray]) -> None:
    """
    Export fields to Legacy VTK format (BINARY version).
    Binary format is faster, smaller, and avoids ASCII parsing errors.
    """
    nx, ny, nz = grid.shape
    dx, dy, dz = grid.spacing
    total_points = nx * ny * nz

    # 注意：必须使用 'wb' (二进制写入模式)
    with path.open("wb") as fh:
        # 1. 写入文件头 (ASCII 字符)
        fh.write(b"# vtk DataFile Version 3.0\n")
        fh.write(b"PhaseFieldSimulation\n")
        fh.write(b"BINARY\n")  # 关键：声明为二进制
        fh.write(b"DATASET STRUCTURED_POINTS\n")
        
        # 格式化字符串需要 encode 为 bytes
        fh.write(f"DIMENSIONS {nx} {ny} {nz}\n".encode('ascii'))
        fh.write(b"ORIGIN 0 0 0\n")
        fh.write(f"SPACING {dx:.6e} {dy:.6e} {dz:.6e}\n".encode('ascii'))
        
        fh.write(f"POINT_DATA {total_points}\n".encode('ascii'))
        
        # 2. 循环写入每个变量
        for name, data in fields.items():
            data_arr = np.asarray(data, dtype=np.float32) # VTK 默认 float 是 32位
            
            # 写入变量头
            fh.write(f"SCALARS {name} float 1\n".encode('ascii'))
            fh.write(b"LOOKUP_TABLE default\n")
            
            # 【关键】数据重排：转置为 x 变化最快 (z, y, x)
            if data_arr.ndim == 3:
                flat_data = data_arr.transpose(2, 1, 0).flatten()
            else:
                flat_data = data_arr.flatten()

            if flat_data.size != total_points:
                print(f"[WARNING] Field '{name}' size mismatch. Skipped.")
                continue
            
            # 【关键】写入二进制数据 (大端序 Big Endian)
            # VTK 标准要求二进制数据必须是 Big Endian
            flat_data.astype('>f4').tofile(fh)
            
            # 二进制块后通常加个换行，虽然不是必须的，但有些读取器需要
            fh.write(b"\n")