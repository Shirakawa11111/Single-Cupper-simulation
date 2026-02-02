"""
Shared I/O helpers for exporting fields to LAMMPS-friendly and VTK formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

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


def write_lammpstrj(
    path: Path,
    grid: GridSpec,
    fields: Dict[str, np.ndarray],
    timestep: int = 0,
    macro_strain: Tuple[float, float, float] | None = None,
) -> None:
    """
    Export fields to LAMMPS trajectory format. If a displacement field is provided,
    coordinates are offset to visualize the deformed configuration. If macro_strain
    is provided, apply affine displacement (eps_x x, eps_y y, eps_z z).
    """
    nx, ny, nz = grid.shape
    dx, dy, dz = grid.spacing
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz
    coords_x = np.linspace(0, Lx - dx, nx)
    coords_y = np.linspace(0, Ly - dy, ny)
    coords_z = np.linspace(0, Lz - dz, nz)
    
    crack = fields.get("crack", np.zeros((nx, ny, nz)))
    accum_plastic = fields.get("accum_plastic", fields.get("plastic", np.zeros((nx, ny, nz))))
    plastic_inst = fields.get("plastic_inst")
    psi = fields.get("psi", np.zeros((nx, ny, nz)))
    plastic_vec = fields.get("plastic_vec")
    stress_vm = fields.get("stress_vm")
    stress = fields.get("stress")
    displacement = fields.get("displacement")
    macro = macro_strain if macro_strain is not None else (0.0, 0.0, 0.0)

    extra_scalars = []
    if plastic_inst is not None and plastic_inst.shape == (nx, ny, nz):
        extra_scalars.append("plastic_inst")
    if plastic_vec is not None and plastic_vec.shape == (nx, ny, nz, 3):
        extra_scalars.extend(["plastic_xx", "plastic_yy", "plastic_zz"])
    stress_scalars = []
    if stress_vm is not None and stress_vm.shape == (nx, ny, nz):
        stress_scalars.append("stress_vm")
    if stress is not None and stress.shape == (nx, ny, nz, 3, 3):
        stress_scalars.extend(["stress_xx", "stress_yy", "stress_zz"])
    
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ITEM: TIMESTEP\n")
        fh.write(f"{timestep}\n")
        fh.write("ITEM: NUMBER OF ATOMS\n")
        fh.write(f"{nx * ny * nz}\n")
        fh.write("ITEM: BOX BOUNDS pp pp pp\n")
        fh.write(f"0.0 {Lx:.6e}\n")
        fh.write(f"0.0 {Ly:.6e}\n")
        fh.write(f"0.0 {Lz:.6e}\n")
        cols = ["id", "type", "x", "y", "z", "crack", "accum_plastic", "psi"] + extra_scalars + stress_scalars
        fh.write("ITEM: ATOMS " + " ".join(cols) + "\n")
        
        atom_id = 1
        for k in range(nz):
            z = coords_z[k]
            for j in range(ny):
                y = coords_y[j]
                for i in range(nx):
                    x = coords_x[i]
                    ux_macro = macro[0] * x
                    uy_macro = macro[1] * y
                    uz_macro = macro[2] * z
                    ux_micro, uy_micro, uz_micro = (displacement[i, j, k] if displacement is not None else (0.0, 0.0, 0.0))
                    x_out = x + ux_macro + ux_micro
                    y_out = y + uy_macro + uy_micro
                    z_out = z + uz_macro + uz_micro
                    values = [
                        atom_id,
                        1,
                        x_out,
                        y_out,
                        z_out,
                        crack[i, j, k],
                        accum_plastic[i, j, k],
                        psi[i, j, k],
                    ]
                    if plastic_inst is not None and plastic_inst.shape == (nx, ny, nz):
                        values.append(plastic_inst[i, j, k])
                    if plastic_vec is not None and plastic_vec.shape == (nx, ny, nz, 3):
                        px, py, pz = plastic_vec[i, j, k]
                        values.extend([px, py, pz])
                    if stress_vm is not None and stress_vm.shape == (nx, ny, nz):
                        values.append(stress_vm[i, j, k])
                    if stress is not None and stress.shape == (nx, ny, nz, 3, 3):
                        sxx = stress[i, j, k, 0, 0]
                        syy = stress[i, j, k, 1, 1]
                        szz = stress[i, j, k, 2, 2]
                        values.extend([sxx, syy, szz])
                    fh.write(" ".join(f"{v:.6e}" if isinstance(v, float) else str(v) for v in values) + "\n")
                    atom_id += 1


def write_vtk(
    path: Path,
    grid: GridSpec,
    fields: Dict[str, np.ndarray],
    macro_strain: Tuple[float, float, float] | None = None,
    deform_coordinates: bool = False,
) -> None:
    """
    Export fields to Legacy VTK format (BINARY version).
    Supports scalar fields (3D) and vector fields (4D with trailing length-3).
    Binary format is faster, smaller, and avoids ASCII parsing errors.
    If deform_coordinates=True and a displacement field is present, writes STRUCTURED_GRID
    with coordinates warped by macro_strain + micro displacement so geometry appears deformed.
    """
    nx, ny, nz = grid.shape
    dx, dy, dz = grid.spacing
    total_points = nx * ny * nz
    scalar_fields = {k: v for k, v in fields.items() if v.ndim == 3 and k != "plastic"}
    vector_fields = {k: v for k, v in fields.items() if v.ndim == 4 and v.shape[-1] == 3}
    if "accum_plastic" not in scalar_fields and "plastic" in fields:
        scalar_fields["accum_plastic"] = fields["plastic"]
    macro = macro_strain if macro_strain is not None else (0.0, 0.0, 0.0)
    disp_field = fields.get("displacement")

    def _sanitize(arr: np.ndarray, clip: float | None = None) -> np.ndarray:
        """Remove NaN/inf and optional clip to keep VTK readable."""
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if clip is not None:
            arr = np.clip(arr, -clip, clip)
        return arr.astype(np.float32, copy=False)

    macro_disp = None
    if disp_field is not None and any(abs(m) > 0 for m in macro):
        mx, my, mz = macro
        macro_disp = np.zeros_like(disp_field, dtype=np.float32)
        coords_x = np.linspace(0, dx * (nx - 1), nx, dtype=np.float32)
        coords_y = np.linspace(0, dy * (ny - 1), ny, dtype=np.float32)
        coords_z = np.linspace(0, dz * (nz - 1), nz, dtype=np.float32)
        macro_disp[..., 0] = mx * coords_x[:, None, None]
        macro_disp[..., 1] = my * coords_y[None, :, None]
        macro_disp[..., 2] = mz * coords_z[None, None, :]
        vector_fields["displacement_total"] = disp_field + macro_disp
    elif disp_field is not None:
        vector_fields["displacement_total"] = disp_field

    # Normalized helper fields for stable colorbars
    if "crack" in scalar_fields:
        crack = scalar_fields["crack"]
        scalar_fields["crack_norm"] = np.clip(crack, 0.0, 1.0)
        scalar_fields["crack_clamp03"] = np.clip(crack, 0.0, 0.3)
    if "accum_plastic" in scalar_fields:
        plastic = scalar_fields["accum_plastic"]
        maxp = np.max(np.abs(plastic)) + 1e-12
        scalar_fields["accum_plastic_norm"] = plastic / maxp
    if "plastic_vec" in vector_fields:
        pv = vector_fields["plastic_vec"]
        mag = np.linalg.norm(pv, axis=-1)
        maxmag = np.max(mag) + 1e-12
        scalar_fields["plastic_vec_mag"] = mag
        scalar_fields["plastic_vec_norm"] = mag / maxmag
        scalar_fields["plastic_vec_x_norm"] = np.clip(pv[..., 0] / (np.max(np.abs(pv[..., 0])) + 1e-12), 0.0, 1.0)
    if "stress_vm" in scalar_fields:
        svm = scalar_fields["stress_vm"]
        maxs = np.max(np.abs(svm)) + 1e-12
        scalar_fields["stress_vm_norm"] = svm / maxs
    if "displacement_total" in vector_fields:
        dtotal = vector_fields["displacement_total"]
        mag = np.linalg.norm(dtotal, axis=-1)
        maxd = np.max(mag) + 1e-12
        scalar_fields["disp_total_mag"] = mag
        scalar_fields["disp_total_norm"] = mag / maxd

    if deform_coordinates and disp_field is not None:
        # Binary STRUCTURED_GRID with deformed coordinates (macro + micro displacement)
        with path.open("wb") as fh:
            fh.write(b"# vtk DataFile Version 3.0\n")
            fh.write(b"PhaseFieldSimulation\n")
            fh.write(b"BINARY\n")
            fh.write(b"DATASET STRUCTURED_GRID\n")
            fh.write(f"DIMENSIONS {nx} {ny} {nz}\n".encode("ascii"))
            fh.write(f"POINTS {total_points} float\n".encode("ascii"))

            # base coordinates
            xs = np.linspace(0, dx * (nx - 1), nx, dtype=np.float32)
            ys = np.linspace(0, dy * (ny - 1), ny, dtype=np.float32)
            zs = np.linspace(0, dz * (nz - 1), nz, dtype=np.float32)
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
            coords = np.stack((X, Y, Z), axis=-1)
            disp_total = disp_field.copy()
            if macro_disp is not None:
                disp_total = disp_total + macro_disp
            coords = coords + disp_total
            coords = _sanitize(coords)
            coords_flat = coords.transpose(2, 1, 0, 3).reshape(-1, 3)
            coords_flat.astype(">f4").tofile(fh)
            fh.write(b"\n")

            fh.write(f"POINT_DATA {total_points}\n".encode("ascii"))
            for name, data in scalar_fields.items():
                data_arr = _sanitize(np.asarray(data))
                fh.write(f"SCALARS {name} float 1\n".encode("ascii"))
                fh.write(b"LOOKUP_TABLE default\n")
                flat_data = data_arr.transpose(2, 1, 0).flatten()
                if flat_data.size != total_points:
                    print(f"[WARNING] Field '{name}' size mismatch. Skipped.")
                    continue
                flat_data.astype(">f4").tofile(fh)
                fh.write(b"\n")
            for name, data in vector_fields.items():
                data_arr = _sanitize(np.asarray(data))
                fh.write(f"VECTORS {name} float\n".encode("ascii"))
                if data_arr.ndim == 4 and data_arr.shape[-1] == 3:
                    flat_data = data_arr.transpose(2, 1, 0, 3).reshape(-1, 3)
                else:
                    flat_data = data_arr.reshape(-1, 3)
                if flat_data.shape[0] != total_points:
                    print(f"[WARNING] Vector field '{name}' size mismatch. Skipped.")
                    continue
                flat_data.astype(">f4").tofile(fh)
                fh.write(b"\n")
    else:
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
            
            # 2. 循环写入标量
            for name, data in scalar_fields.items():
                data_arr = _sanitize(np.asarray(data))
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

            # 3. 循环写入向量场（如位移）
            for name, data in vector_fields.items():
                data_arr = _sanitize(np.asarray(data))
                fh.write(f"VECTORS {name} float\n".encode('ascii'))
                if data_arr.ndim == 4 and data_arr.shape[-1] == 3:
                    flat_data = data_arr.transpose(2, 1, 0, 3).reshape(-1, 3)
                else:
                    flat_data = data_arr.reshape(-1, 3)
                if flat_data.shape[0] != total_points:
                    print(f"[WARNING] Vector field '{name}' size mismatch. Skipped.")
                    continue
                flat_data.astype('>f4').tofile(fh)
                fh.write(b"\n")
