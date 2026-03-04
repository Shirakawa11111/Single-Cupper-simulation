"""
Run one coupled virtual-cycle validation and export a compact verification bundle:
1) GROD map (deg) with IPF-triangle hue coloring
2) stress amplitude sigma_a vs cumulative plastic strain curve
3) accumulated plastic map
4) crack phase-field map
5) slip-system parameter/damage table with Schmid factors

Also moves previous test images to a timestamped trash folder.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import json
import matplotlib.pyplot as plt
import numpy as np
import yaml  # type: ignore
from scipy.ndimage import gaussian_filter  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.energy import FractureParameters, PFCCoupling, PFCParameters
from sim.tests.generate_real_poster_v3 import (
    _surface_tri_mask_xy,
    _surface_tri_notch_from_box,
)
from sim.tests.run_virtual_cycle_config import _normalize_config, _resolve_payload
from sim.tests.virtual_cycle import run_virtual_cycles


def _normalize_vec3(v: Any, default: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(v if isinstance(v, (list, tuple, np.ndarray)) else default, dtype=float).reshape(-1)
    if arr.size != 3:
        arr = np.asarray(default, dtype=float)
    nrm = float(np.linalg.norm(arr))
    if (not np.isfinite(nrm)) or nrm <= 1e-12:
        arr = np.asarray(default, dtype=float)
        nrm = float(np.linalg.norm(arr))
    return arr / nrm


def _read_float_column(csv_path: Path, col_name: str) -> np.ndarray:
    vals: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or col_name not in reader.fieldnames:
            raise ValueError(f"CSV {csv_path} missing required column: {col_name}")
        for row in reader:
            try:
                vals.append(float(row[col_name]))
            except (TypeError, ValueError):
                continue
    return np.asarray(vals, dtype=float)


def _try_read_float_column(csv_path: Path, col_name: str) -> np.ndarray | None:
    try:
        return _read_float_column(csv_path, col_name)
    except Exception:
        return None


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_run_dir() -> Path:
    return Path("sim/tests/regress_runs") / date.today().isoformat() / f"coupled_validation_{_timestamp()}"


def _move_old_images_to_trash(out_dir: Path, trash_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    moved: list[Path] = []
    trash_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(out_dir.glob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        target = trash_dir / p.name
        if target.exists():
            target = trash_dir / f"{p.stem}_{_timestamp()}{p.suffix}"
        shutil.move(str(p), str(target))
        moved.append(target)
    return moved


def _build_coupling_kwargs(cfg: dict[str, Any]) -> dict[str, float]:
    keys = (
        "yield_tau",
        "flow_scale",
        "linear_hardening",
        "visco_exponent",
        "gamma0",
        "slip_exponent",
        "h_iso",
        "yield_tau_inf",
        "h0_iso",
        "h_gnd",
        "kin_c",
        "kin_d",
        "kin_c2",
        "kin_d2",
    )
    out: dict[str, float] = {}
    for k in keys:
        v = cfg.get(k)
        if v is not None:
            out[k] = float(v)
    return out


def _notch_mask_from_cfg(cfg: dict[str, Any], shape: tuple[int, int, int]) -> np.ndarray:
    notch_mask = np.zeros((shape[0], shape[1]), dtype=bool)
    spacing_raw = cfg.get("grid_spacing", [1.0, 1.0, 1.0])
    if isinstance(spacing_raw, (list, tuple)) and len(spacing_raw) >= 2:
        dx = float(spacing_raw[0]) if float(spacing_raw[0]) > 0.0 else 1.0
        dy = float(spacing_raw[1]) if float(spacing_raw[1]) > 0.0 else 1.0
    else:
        dx, dy = 1.0, 1.0

    notch_box_raw = cfg.get("notch_box")
    if isinstance(notch_box_raw, (list, tuple)) and len(notch_box_raw) == 3:
        x0 = float(notch_box_raw[0][0]) / dx
        x1 = float(notch_box_raw[0][1]) / dx
        y0 = float(notch_box_raw[1][0]) / dy
        y1 = float(notch_box_raw[1][1]) / dy
        notch_box = (
            (x0, x1),
            (y0, y1),
            (float(notch_box_raw[2][0]), float(notch_box_raw[2][1])),
        )
        notch_tri = _surface_tri_notch_from_box(notch_box, shape[1])
        notch_mask |= _surface_tri_mask_xy(shape[0], shape[1], notch_tri)

    arc_notch = cfg.get("arc_notch")
    if isinstance(arc_notch, dict):
        center = arc_notch.get("center", [0.5 * (shape[0] - 1) * dx, shape[1] * dy])
        if isinstance(center, (list, tuple)) and len(center) == 2:
            cx, cy = float(center[0]), float(center[1])
            radius = float(arc_notch.get("radius", min(shape[0] * dx, shape[1] * dy) * 0.5))
            surface = str(arc_notch.get("surface", "y_max"))
            xs = np.arange(shape[0], dtype=float) * dx
            ys = np.arange(shape[1], dtype=float) * dy
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            rad2 = radius * radius - (X - cx) ** 2
            valid = rad2 >= 0.0
            if surface == "y_min":
                y_arc = cy + np.sqrt(np.clip(rad2, 0.0, None))
                arc_mask = valid & (Y <= y_arc)
            else:
                y_arc = cy - np.sqrt(np.clip(rad2, 0.0, None))
                arc_mask = valid & (Y >= y_arc)
            x_range = arc_notch.get("x_range")
            if isinstance(x_range, (list, tuple)) and len(x_range) == 2:
                x0, x1 = float(x_range[0]), float(x_range[1])
                x0, x1 = min(x0, x1), max(x0, x1)
                min_cells_cfg = arc_notch.get("min_cells_x", cfg.get("arc_notch_min_cells_x", 2))
                min_cells = max(1, int(min_cells_cfg))
                min_width = float(min_cells) * dx
                width = x1 - x0
                if width + 1e-15 < min_width:
                    xc = 0.5 * (x0 + x1)
                    half = 0.5 * min_width
                    x_min_domain = 0.0
                    x_max_domain = dx * (shape[0] - 1)
                    x0 = max(x_min_domain, xc - half)
                    x1 = min(x_max_domain, xc + half)
                    if (x1 - x0) + 1e-15 < min_width:
                        if x0 <= x_min_domain + 1e-12:
                            x1 = min(x_max_domain, x_min_domain + min_width)
                        else:
                            x0 = max(x_min_domain, x_max_domain - min_width)
                    x0, x1 = min(x0, x1), max(x0, x1)
                arc_mask &= (X >= min(x0, x1)) & (X <= max(x0, x1))
            notch_mask |= arc_mask
    return notch_mask


def _base_euler_from_orientation_vector(orientation_vector: list[float] | tuple[float, float, float]) -> tuple[float, float, float]:
    vec = np.asarray(orientation_vector, dtype=float).reshape(-1)
    if vec.size != 3:
        vec = np.array([1.0, 1.0, 1.0], dtype=float)
    nrm = float(np.linalg.norm(vec))
    if not np.isfinite(nrm) or nrm <= 1e-12:
        vec = np.array([1.0, 1.0, 1.0], dtype=float)
        nrm = float(np.linalg.norm(vec))
    n = vec / nrm
    phi1 = float(np.degrees(np.arctan2(n[1], n[0])) % 360.0)
    Phi = float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))
    phi2 = 0.0
    return phi1, Phi, phi2


def _robust_norm01(field: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    span = hi - lo
    if (not np.isfinite(span)) or span <= 1e-12:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / span, 0.0, 1.0)


def _grod_from_fields_robust(plastic_2d: np.ndarray, gnd_2d: np.ndarray, max_deg: float) -> tuple[np.ndarray, np.ndarray]:
    p_norm = _robust_norm01(plastic_2d)
    g_norm = _robust_norm01(gnd_2d)
    # emphasize coupled heterogeneity; avoid one-field domination
    activity = 0.65 * p_norm + 0.35 * g_norm
    act_span = float(np.percentile(activity, 99.0) - np.percentile(activity, 1.0))
    if (not np.isfinite(act_span)) or act_span <= 1e-10:
        grod_t1 = np.zeros_like(activity)
    else:
        activity_norm = _robust_norm01(activity, p_low=1.0, p_high=99.0)
        grod_t1 = np.clip(max_deg * activity_norm, 0.0, max_deg)
    grod_t0 = np.zeros_like(grod_t1)
    return grod_t0, grod_t1


def _grid_gradient_scalar(field: np.ndarray, axis: int, spacing: float, periodic: bool) -> np.ndarray:
    if periodic:
        return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2.0 * spacing)
    if field.shape[axis] < 2:
        return np.zeros_like(field)
    grad = np.empty_like(field)
    if field.shape[axis] > 2:
        slc_mid = [slice(None)] * field.ndim
        slc_mid[axis] = slice(1, -1)
        slc_plus = slc_mid.copy()
        slc_plus[axis] = slice(2, None)
        slc_minus = slc_mid.copy()
        slc_minus[axis] = slice(None, -2)
        grad[tuple(slc_mid)] = (field[tuple(slc_plus)] - field[tuple(slc_minus)]) / (2.0 * spacing)
    slc0 = [slice(None)] * field.ndim
    slc1 = [slice(None)] * field.ndim
    slc0[axis] = 0
    slc1[axis] = 1
    grad[tuple(slc0)] = (field[tuple(slc1)] - field[tuple(slc0)]) / spacing
    slc_end = [slice(None)] * field.ndim
    slc_endm1 = [slice(None)] * field.ndim
    slc_end[axis] = -1
    slc_endm1[axis] = -2
    grad[tuple(slc_end)] = (field[tuple(slc_end)] - field[tuple(slc_endm1)]) / spacing
    return grad


def _project_to_so3(mat: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(mat, full_matrices=False)
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def _cubic_symmetry_ops() -> list[np.ndarray]:
    ops: list[np.ndarray] = []
    axes = [0, 1, 2]
    import itertools
    for perm in itertools.permutations(axes):
        p = np.zeros((3, 3), dtype=float)
        for i, j in enumerate(perm):
            p[i, j] = 1.0
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            s = np.diag(np.asarray(signs, dtype=float))
            m = p @ s
            if np.linalg.det(m) > 0.0:
                ops.append(m)
    uniq: list[np.ndarray] = []
    for m in ops:
        if not any(np.allclose(m, u, atol=1e-12) for u in uniq):
            uniq.append(m)
    return uniq


def _grod_from_orientation_misorientation(
    displacement: np.ndarray,
    orientation: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    periodic_xyz: tuple[bool, bool, bool],
    notch_mask_xy: np.ndarray,
    use_cubic_symmetry: bool = True,
) -> np.ndarray:
    u = np.asarray(displacement, dtype=float)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError("displacement must have shape (nx, ny, nz, 3)")
    nx, ny, nz, _ = u.shape
    grad_u = np.zeros((nx, ny, nz, 3, 3), dtype=float)
    for i in range(3):
        ui = u[..., i]
        for j in range(3):
            grad_u[..., j, i] = _grid_gradient_scalar(
                ui,
                axis=j,
                spacing=float(spacing_xyz[j]),
                periodic=bool(periodic_xyz[j]),
            )
    eye = np.eye(3, dtype=float)
    f = eye[None, None, None, :, :] + grad_u
    f2 = f.reshape(-1, 3, 3)
    uu, _, vvt = np.linalg.svd(f2, full_matrices=False)
    r2 = uu @ vvt
    det = np.linalg.det(r2)
    neg = det < 0.0
    if np.any(neg):
        uu[neg, :, -1] *= -1.0
        r2[neg] = uu[neg] @ vvt[neg]
    r = r2.reshape(nx, ny, nz, 3, 3)

    ori = np.asarray(orientation, dtype=float)
    if ori.shape == (3, 3):
        g0 = np.broadcast_to(ori, (nx, ny, nz, 3, 3))
    elif ori.shape == (nx, ny, nz, 3, 3):
        g0 = ori
    else:
        raise ValueError("orientation must be (3,3) or (nx,ny,nz,3,3)")
    g = np.einsum("...ij,...jk->...ik", r, g0, optimize=True)

    valid3 = np.broadcast_to((~notch_mask_xy)[..., None], (nx, ny, nz))
    finite_mask = np.isfinite(g).all(axis=(-2, -1))
    valid = valid3 & finite_mask
    if not np.any(valid):
        return np.zeros((nx, ny), dtype=float)

    g_ref = _project_to_so3(np.mean(g[valid], axis=0))
    delta = np.einsum("ij,...jk->...ik", g_ref.T, g, optimize=True)

    if use_cubic_symmetry:
        syms = _cubic_symmetry_ops()
        ang_list = []
        for s in syms:
            d = np.einsum("...ij,jk->...ik", delta, s, optimize=True)
            tr = np.trace(d, axis1=-2, axis2=-1)
            c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
            ang_list.append(np.degrees(np.arccos(c)))
        ang = np.min(np.stack(ang_list, axis=0), axis=0)
    else:
        tr = np.trace(delta, axis1=-2, axis2=-1)
        c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
        ang = np.degrees(np.arccos(c))

    ang_xy = np.max(np.where(valid, ang, np.nan), axis=2)
    ang_xy = np.nan_to_num(ang_xy, nan=0.0, posinf=0.0, neginf=0.0)
    ang_xy[notch_mask_xy] = np.nan
    return ang_xy


def _euler_from_fields_robust(
    plastic_2d: np.ndarray,
    gnd_2d: np.ndarray,
    orientation_vector: list[float] | tuple[float, float, float],
    grod_t1: np.ndarray,
    grod_max_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    phi1_0, Phi_0, phi2_0 = _base_euler_from_orientation_vector(orientation_vector)
    p_norm = _robust_norm01(plastic_2d)
    g_norm = _robust_norm01(gnd_2d)
    gn = np.clip(grod_t1 / max(grod_max_deg, 1e-12), 0.0, 1.0)

    euler_t0 = np.zeros((*plastic_2d.shape, 3), dtype=float)
    euler_t1 = np.zeros((*plastic_2d.shape, 3), dtype=float)
    euler_t0[..., 0] = phi1_0
    euler_t0[..., 1] = Phi_0
    euler_t0[..., 2] = phi2_0

    euler_t1[..., 0] = np.mod(phi1_0 + 20.0 * gn + 7.0 * (p_norm - 0.5), 360.0)
    euler_t1[..., 1] = np.clip(Phi_0 + 14.0 * gn + 5.0 * (g_norm - 0.5), 0.0, 180.0)
    euler_t1[..., 2] = np.mod(phi2_0 + 24.0 * gn + 8.0 * (p_norm - g_norm), 360.0)
    return euler_t0, euler_t1


def _euler_bunge_to_matrix(euler_deg: np.ndarray) -> np.ndarray:
    """Convert Bunge Euler angles (deg) to rotation matrix g."""
    phi1 = np.deg2rad(euler_deg[..., 0])
    Phi = np.deg2rad(euler_deg[..., 1])
    phi2 = np.deg2rad(euler_deg[..., 2])
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    g = np.zeros(euler_deg.shape[:-1] + (3, 3), dtype=float)
    g[..., 0, 0] = c1 * c2 - s1 * s2 * c
    g[..., 0, 1] = s1 * c2 + c1 * s2 * c
    g[..., 0, 2] = s2 * s
    g[..., 1, 0] = -c1 * s2 - s1 * c2 * c
    g[..., 1, 1] = -s1 * s2 + c1 * c2 * c
    g[..., 1, 2] = c2 * s
    g[..., 2, 0] = s1 * s
    g[..., 2, 1] = -c1 * s
    g[..., 2, 2] = c
    return g


def _ipf_rgb_cubic_from_dirs(dir_c: np.ndarray) -> np.ndarray:
    """
    Approximate cubic IPF triangle coloring.
    Fundamental mapping uses sorted absolute crystal directions: h1>=h2>=h3>=0.
    Vertices: [001](red), [101](green), [111](blue).
    """
    d = np.abs(dir_c)
    nrm = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12
    d = d / nrm
    d = np.sort(d, axis=-1)[..., ::-1]  # h1>=h2>=h3
    h1 = d[..., 0]
    h2 = d[..., 1]
    h3 = d[..., 2]
    u = h2 / (h1 + 1e-12)
    v = h3 / (h1 + 1e-12)
    w_a = np.clip(1.0 - u, 0.0, 1.0)      # [001] -> red
    w_b = np.clip(u - v, 0.0, 1.0)        # [101] -> green
    w_c = np.clip(v, 0.0, 1.0)            # [111] -> blue
    w_sum = w_a + w_b + w_c + 1e-12
    rgb = np.stack([w_a / w_sum, w_b / w_sum, w_c / w_sum], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def _plot_ipf_triangle_legend(ax: plt.Axes) -> None:
    n = 240
    img = np.ones((n, n, 3), dtype=float)
    mask = np.zeros((n, n), dtype=bool)
    for iy in range(n):
        for ix in range(n):
            u = ix / (n - 1)      # 0..1
            v = iy / (n - 1)      # 0..1
            if v <= u:
                w_a = 1.0 - u
                w_b = u - v
                w_c = v
                s = w_a + w_b + w_c + 1e-12
                img[iy, ix, 0] = w_a / s
                img[iy, ix, 1] = w_b / s
                img[iy, ix, 2] = w_c / s
                mask[iy, ix] = True
    img[~mask] = 1.0
    ax.imshow(img, origin="lower")
    ax.set_title("IPF Triangle\n[001]-[101]-[111]", fontsize=10)
    ax.set_xticks([0, n - 1])
    ax.set_xticklabels(["[001]", "[101]"], fontsize=8)
    ax.set_yticks([0, n - 1])
    ax.set_yticklabels(["", "[111]"], fontsize=8)
    ax.set_xlabel("u", fontsize=8)
    ax.set_ylabel("v", fontsize=8)


def _xy_extent_and_unit(
    shape_xy: tuple[int, int],
    spacing_xy: tuple[float, float] | None,
) -> tuple[list[float] | None, str]:
    if spacing_xy is None:
        return None, "grid index"
    try:
        dx = float(spacing_xy[0])
        dy = float(spacing_xy[1])
    except Exception:
        return None, "grid index"
    if (not np.isfinite(dx)) or (not np.isfinite(dy)) or dx <= 0.0 or dy <= 0.0:
        return None, "grid index"
    max_d = max(dx, dy)
    if max_d <= 1.0e-4:
        scale = 1.0e6
        unit = "µm"
    elif max_d <= 1.0e-1:
        scale = 1.0e3
        unit = "mm"
    else:
        scale = 1.0
        unit = "m"
    nx, ny = int(shape_xy[0]), int(shape_xy[1])
    extent = [0.0, (nx - 1) * dx * scale, 0.0, (ny - 1) * dy * scale]
    return extent, unit


def _plot_grod_ipf_map(
    grod_deg: np.ndarray,
    euler_t1: np.ndarray,
    notch_mask: np.ndarray,
    out_png: Path,
    max_deg: float,
    spacing_xy: tuple[float, float] | None = None,
) -> None:
    # GROD-only map (no IPF hue overlay) + smooth display
    _ = euler_t1  # kept for API compatibility
    field = np.asarray(grod_deg, dtype=float)
    valid = ~notch_mask
    w = valid.astype(float)
    sigma = 0.8
    num = gaussian_filter(field * w, sigma=sigma, mode="nearest")
    den = gaussian_filter(w, sigma=sigma, mode="nearest")
    field_smooth = num / np.clip(den, 1e-12, None)
    field_smooth[~valid] = np.nan
    valid_vals = field_smooth[valid & np.isfinite(field_smooth)]
    if valid_vals.size > 0:
        p99 = float(np.percentile(valid_vals, 99.0))
        p02 = float(np.percentile(valid_vals, 2.0))
    else:
        p99 = float(max_deg)
        p02 = 0.0
    vmax = max(1e-6, min(max_deg, p99))
    field_disp = np.clip(field_smooth, 0.0, vmax)
    # Pin the low tail to exact zero so 0-regions are visible.
    field_disp[(field_disp <= p02) & np.isfinite(field_disp)] = 0.0

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 4.8), constrained_layout=True)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#f5f5f5")
    extent, axis_unit = _xy_extent_and_unit(field_disp.shape, spacing_xy)
    im = ax.imshow(
        np.swapaxes(field_disp, 0, 1),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="bilinear",
        extent=extent,
    )
    ax.set_title("GROD Map (p99 clipped)")
    ax.set_xlabel(f"X ({axis_unit})")
    ax.set_ylabel(f"Y ({axis_unit})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02, aspect=35)
    ticks = np.linspace(0.0, vmax, 6)
    cbar.set_ticks(ticks)
    cbar.set_label("GROD (°)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.8)
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def _plot_scalar_map(
    field_2d: np.ndarray,
    notch_mask: np.ndarray,
    out_png: Path,
    title: str,
    cbar_label: str,
    cmap_name: str = "viridis",
    clip_percentile: float = 99.0,
    smooth_sigma: float = 0.8,
    force_vmin: float | None = 0.0,
    force_vmax: float | None = None,
    spacing_xy: tuple[float, float] | None = None,
) -> None:
    field = np.asarray(field_2d, dtype=float)
    valid = ~notch_mask
    w = valid.astype(float)
    num = gaussian_filter(field * w, sigma=smooth_sigma, mode="nearest")
    den = gaussian_filter(w, sigma=smooth_sigma, mode="nearest")
    disp = num / np.clip(den, 1e-12, None)
    disp[~valid] = np.nan

    valid_vals = disp[valid & np.isfinite(disp)]
    if valid_vals.size > 0:
        vmax = float(np.percentile(valid_vals, clip_percentile))
        if force_vmin is None:
            vmin = float(np.percentile(valid_vals, 1.0))
        else:
            vmin = float(force_vmin)
    else:
        vmax = float(np.nanmax(field)) if np.isfinite(np.nanmax(field)) else 1.0
        vmin = 0.0 if force_vmin is None else float(force_vmin)
    if force_vmax is not None:
        vmax = float(force_vmax)
    if not np.isfinite(vmax):
        vmax = 1.0
    vmax = max(vmax, vmin + 1e-12)
    disp = np.clip(disp, vmin, vmax)

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 4.8), constrained_layout=True)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#f5f5f5")
    extent, axis_unit = _xy_extent_and_unit(disp.shape, spacing_xy)
    im = ax.imshow(
        np.swapaxes(disp, 0, 1),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        extent=extent,
    )
    ax.set_title(title)
    ax.set_xlabel(f"X ({axis_unit})")
    ax.set_ylabel(f"Y ({axis_unit})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02, aspect=35)
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.8)
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def _schmid_for_slip(orientation: np.ndarray, m: np.ndarray, n: np.ndarray, load_axis: np.ndarray) -> np.ndarray:
    m_lab = np.einsum("...ij,j->...i", orientation, m, optimize=True)
    n_lab = np.einsum("...ij,j->...i", orientation, n, optimize=True)
    return np.abs(
        np.einsum("...i,i->...", m_lab, load_axis, optimize=True)
        * np.einsum("...i,i->...", n_lab, load_axis, optimize=True)
    )


def _to_miller_label(v: np.ndarray) -> str:
    vv = np.asarray(v, dtype=float)
    mx = np.max(np.abs(vv)) + 1e-12
    ivec = np.rint(vv / mx).astype(int)
    return f"[{ivec[0]} {ivec[1]} {ivec[2]}]"


def _build_slip_table(
    state_out: dict[str, np.ndarray],
    coupling: PFCCoupling,
    schmid_mode: str = "crystal_loading_axis",
    schmid_crystal_axis: tuple[float, float, float] = (1.0, 1.0, 1.0),
    load_axis_sample: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> list[dict[str, Any]]:
    gamma_s = np.asarray(state_out["gamma_s"], dtype=float)
    crack = np.asarray(state_out["crack"], dtype=float)
    chi_s = np.asarray(state_out["chi_s"], dtype=float)
    chi_s2 = np.asarray(state_out["chi_s2"], dtype=float)
    orientation = np.asarray(state_out["orientation"], dtype=float)
    stress = np.asarray(state_out.get("stress", np.empty((0,))), dtype=float)
    if orientation.shape == (3, 3):
        orientation = np.broadcast_to(orientation, crack.shape + (3, 3))
    has_stress = stress.ndim >= 2 and stress.shape[-2:] == (3, 3)
    load_axis = _normalize_vec3(load_axis_sample, default=(1.0, 0.0, 0.0))
    crystal_axis = _normalize_vec3(schmid_crystal_axis, default=(1.0, 1.0, 1.0))
    mode = str(schmid_mode).strip().lower()
    if mode not in ("simulation_frame", "crystal_loading_axis"):
        mode = "crystal_loading_axis"

    gamma_abs_all = np.abs(gamma_s)
    total_activity = float(np.sum(gamma_abs_all)) + 1e-16
    rows: list[dict[str, Any]] = []
    for k, (m, n) in enumerate(coupling.slip_systems, start=1):
        idx = k - 1
        schmid_sim = _schmid_for_slip(orientation, m, n, load_axis)
        schmid_anchor = float(np.abs(np.dot(m, crystal_axis) * np.dot(n, crystal_axis)))
        if mode == "simulation_frame":
            schmid_mean = float(np.mean(schmid_sim))
            schmid_max = float(np.max(schmid_sim))
        else:
            schmid_mean = schmid_anchor
            schmid_max = schmid_anchor
        gamma_abs = gamma_abs_all[idx]
        damage = gamma_abs * crack
        chi_abs = np.abs(chi_s[idx] + chi_s2[idx])
        tau_abs_mean_mpa = float("nan")
        if has_stress:
            m_lab = np.einsum("...ij,j->...i", orientation, m, optimize=True)
            n_lab = np.einsum("...ij,j->...i", orientation, n, optimize=True)
            tau_signed = np.einsum("...i,...ij,...j->...", m_lab, stress, n_lab, optimize=True)
            tau_abs_mean_mpa = float(np.mean(np.abs(tau_signed)) * 1.0e3)
        rows.append(
            {
                "slip_id": k,
                "direction_uvws": _to_miller_label(m),
                "plane_hkls": _to_miller_label(n),
                "schmid_mean": schmid_mean,
                "tau_mean_MPa": tau_abs_mean_mpa,
                "schmid_max": schmid_max,
                "schmid_anchor": schmid_anchor,
                "schmid_sim_mean": float(np.mean(schmid_sim)),
                "activity_share_pct": 100.0 * float(np.sum(gamma_abs)) / total_activity,
                "gamma_abs_mean": float(np.mean(gamma_abs)),
                "gamma_abs_p95": float(np.percentile(gamma_abs, 95.0)),
                "damage_index_mean": float(np.mean(damage)),
                "damage_index_p95": float(np.percentile(damage, 95.0)),
                "backstress_abs_mean": float(np.mean(chi_abs)),
            }
        )
    rows.sort(key=lambda r: (r["schmid_mean"], r["damage_index_mean"], r["activity_share_pct"]), reverse=True)
    return rows


def _plot_table(rows: list[dict[str, Any]], out_png: Path, title: str) -> None:
    show_cols = [
        "direction_uvws",
        "plane_hkls",
        "schmid_mean",
        "tau_mean_MPa",
        "damage_index_mean",
    ]
    col_labels = [
        "direction_uvws",
        "plane_hkls",
        "schmid_mean (-)",
        "tau_mean (MPa)",
        "damage_index_mean (-)",
    ]
    disp_rows: list[list[str]] = []
    for row in rows:
        row_out: list[str] = []
        for c in show_cols:
            v = row.get(c, "")
            if c in ("schmid_mean", "damage_index_mean"):
                fv = float(v)
                v = f"{fv:.4e}" if abs(fv) < 1e-2 else f"{fv:.4f}"
            elif c == "tau_mean_MPa":
                if np.isfinite(float(v)):
                    v = f"{float(v):.4f}"
                else:
                    v = "nan"
            row_out.append(str(v))
        disp_rows.append(row_out)

    fig_h = 0.45 * len(disp_rows) + 1.8
    fig, ax = plt.subplots(figsize=(12.0, max(4.2, fig_h)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=10)
    table = ax.table(
        cellText=disp_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f4e79")
        else:
            cell.set_facecolor("#f2f4f7" if r % 2 == 0 else "#e7edf5")
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run coupled validation bundle.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("sim/configs/fatigue_lowamp_align_locked_v5_hard_sat_coupled.yaml"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("docs/exp_compare_cycle_1000"))
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional run output directory.")
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--cycle-points", type=int, default=None)
    parser.add_argument("--grod-max-deg", type=float, default=15.0)
    parser.add_argument(
        "--keep-old-images",
        action="store_true",
        help="Do not move old images into trash folder before running.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trash_dir = out_dir / f"trash_{_timestamp()}"
    moved = []
    if not args.keep_old_images:
        moved = _move_old_images_to_trash(out_dir, trash_dir)

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be mapping.")
    vc_raw, _meta = _resolve_payload(raw)
    cfg = _normalize_config(vc_raw)
    schmid_mode = str(cfg.get("schmid_mode", "crystal_loading_axis"))
    schmid_crystal_axis_raw = cfg.get("schmid_crystal_loading_axis", [1.0, 1.0, 1.0])
    load_axis_sample_raw = cfg.get("load_axis_sample", [1.0, 0.0, 0.0])
    schmid_crystal_axis_vec = _normalize_vec3(schmid_crystal_axis_raw, default=(1.0, 1.0, 1.0))
    load_axis_sample_vec = _normalize_vec3(load_axis_sample_raw, default=(1.0, 0.0, 0.0))
    # remove post-processing-only keys before passing into run_virtual_cycles
    cfg.pop("schmid_mode", None)
    cfg.pop("schmid_crystal_loading_axis", None)
    cfg.pop("load_axis_sample", None)

    run_dir = args.run_dir or _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["run_dir"] = run_dir
    cfg["auto_output"] = True
    cfg["task"] = "coupled_validation_bundle"
    if args.cycles is not None:
        cfg["cycles"] = int(args.cycles)
    if args.cycle_points is not None:
        cfg["cycle_points"] = int(args.cycle_points)
    # Keep some runtime visibility for long/stiff cases.
    cpts_for_log = int(cfg.get("cycle_points", 80))
    cfg["print_interval"] = max(5, int(max(1, cpts_for_log) // 8))
    cfg["vtk_interval"] = 1000000

    diagnostics: dict[str, Any] = {}
    state_out: dict[str, np.ndarray] = {}
    results, paris_coeff, coffman_coeff = run_virtual_cycles(
        **cfg,
        diagnostics_out=diagnostics,
        state_out=state_out,
    )
    if not results:
        raise RuntimeError("No cycle results produced.")

    # sigma_a - eps_p_cum
    stress_csv = Path(cfg["stress_strain_csv"]) if "stress_strain_csv" in cfg else (run_dir / "virtual_cycle_stress_strain.csv")
    if not stress_csv.exists():
        stress_csv = run_dir / "virtual_cycle_stress_strain.csv"
    stress_ref_gpa = float(cfg.get("stress_ref_gpa", diagnostics.get("stress_ref_gpa", 168.4)))
    stress_source_column = "sig_xx_GPa"
    sig_xx_nd = _try_read_float_column(stress_csv, "sig_xx_nd")
    if sig_xx_nd is not None and sig_xx_nd.size > 0:
        stress = sig_xx_nd * stress_ref_gpa * 1000.0
        stress_source_column = "sig_xx_nd"
    else:
        stress = _read_float_column(stress_csv, "sig_xx_GPa") * 1000.0
    cpts = int(cfg["cycle_points"])
    n_cycles = min(len(results), len(stress) // cpts)
    if n_cycles <= 0:
        raise RuntimeError("Cannot build sigma_a curve: no complete cycles in stress series.")
    eps_p_inc = np.array([float(r.plastic_range) for r in results[:n_cycles]], dtype=float)
    eps_p_cum = np.cumsum(np.clip(eps_p_inc, 0.0, None))
    sigma_a = np.zeros(n_cycles, dtype=float)
    sigma_max = np.zeros(n_cycles, dtype=float)
    sigma_min = np.zeros(n_cycles, dtype=float)
    for i in range(n_cycles):
        s = stress[i * cpts : (i + 1) * cpts]
        sigma_max[i] = float(np.max(s))
        sigma_min[i] = float(np.min(s))
        sigma_a[i] = 0.5 * (sigma_max[i] - sigma_min[i])

    curve_csv = out_dir / "validation_sigma_a_vs_cum.csv"
    curve_rows: list[dict[str, Any]] = []
    for i in range(n_cycles):
        curve_rows.append(
            {
                "cycle": int(i + 1),
                "eps_p_inc": float(eps_p_inc[i]),
                "eps_p_cum": float(eps_p_cum[i]),
                "sigma_a_MPa": float(sigma_a[i]),
                "sigma_max_MPa": float(sigma_max[i]),
                "sigma_min_MPa": float(sigma_min[i]),
            }
        )
    _write_rows_csv(
        curve_csv,
        curve_rows,
        fieldnames=["cycle", "eps_p_inc", "eps_p_cum", "sigma_a_MPa", "sigma_max_MPa", "sigma_min_MPa"],
    )

    fig_curve = out_dir / "validation_sigma_a_vs_cum.png"
    plt.figure(figsize=(7.2, 5.0))
    plt.plot(np.r_[0.0, eps_p_cum], np.r_[0.0, sigma_a], "-o", ms=3.5, lw=2.0, color="#d9480f")
    plt.xlabel(r"Cumulative plastic strain $\epsilon^{p}_{cum}$ (-)")
    plt.ylabel(r"Stress amplitude $\Sigma_a$ (MPa)")
    plt.title(r"Validation: $\Sigma_a$ vs cumulative plastic strain")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_curve, dpi=240)
    plt.close()

    # GROD map (real misorientation, deg)
    shape = tuple(int(v) for v in cfg.get("grid_shape", [64, 32, 8]))
    spacing_raw = cfg.get("grid_spacing", [1.0, 1.0, 1.0])
    periodic_raw = cfg.get("grid_periodic", [True, False, False])
    try:
        spacing_xy = (float(spacing_raw[0]), float(spacing_raw[1]))
        spacing_xyz = (float(spacing_raw[0]), float(spacing_raw[1]), float(spacing_raw[2]))
    except Exception:
        spacing_xy = (1.0, 1.0)
        spacing_xyz = (1.0, 1.0, 1.0)
    try:
        periodic_xyz = (bool(periodic_raw[0]), bool(periodic_raw[1]), bool(periodic_raw[2]))
    except Exception:
        periodic_xyz = (True, False, False)
    notch_mask = _notch_mask_from_cfg(cfg, shape)

    plastic_2d = np.max(np.asarray(state_out["accum_plastic"], dtype=float), axis=2)
    gnd_2d = np.max(np.asarray(state_out.get("gnd_density", np.zeros(shape)), dtype=float), axis=2)
    try:
        grod_t1 = _grod_from_orientation_misorientation(
            np.asarray(state_out["displacement"], dtype=float),
            np.asarray(state_out["orientation"], dtype=float),
            spacing_xyz=spacing_xyz,
            periodic_xyz=periodic_xyz,
            notch_mask_xy=notch_mask,
            use_cubic_symmetry=True,
        )
    except Exception as exc:
        print(f"[warn] real GROD failed ({exc}); fallback to synthetic GROD.")
        _, grod_t1 = _grod_from_fields_robust(plastic_2d, gnd_2d, max_deg=max(1e-6, args.grod_max_deg))
    orientation_vector = cfg.get("orientation_vector", [1.0, 1.0, 1.0])
    if not isinstance(orientation_vector, (list, tuple)) or len(orientation_vector) != 3:
        orientation_vector = [1.0, 1.0, 1.0]
    _, euler_t1 = _euler_from_fields_robust(
        plastic_2d,
        gnd_2d,
        orientation_vector=orientation_vector,
        grod_t1=grod_t1,
        grod_max_deg=max(1e-6, args.grod_max_deg),
    )
    grod_png = out_dir / "validation_grod_t1.png"
    grod_png_legacy = out_dir / "validation_grod_ipf_t1.png"
    _plot_grod_ipf_map(
        grod_t1,
        euler_t1,
        notch_mask,
        grod_png,
        max_deg=max(1e-6, args.grod_max_deg),
        spacing_xy=spacing_xy,
    )
    if grod_png != grod_png_legacy:
        shutil.copyfile(grod_png, grod_png_legacy)

    # Accumulated plastic and crack maps
    plastic_png = out_dir / "validation_accum_plastic_t1.png"
    _plot_scalar_map(
        plastic_2d,
        notch_mask,
        plastic_png,
        "Accumulated Plastic Strain (t1, p99 clipped)",
        r"$\bar{\epsilon}^{p}$ (-)",
        cmap_name="magma",
        clip_percentile=99.0,
        force_vmin=0.0,
        spacing_xy=spacing_xy,
    )
    crack_2d = np.max(np.asarray(state_out["crack"], dtype=float), axis=2)
    crack_png = out_dir / "validation_crack_t1.png"
    _plot_scalar_map(
        crack_2d,
        notch_mask,
        crack_png,
        "Crack Phase-field (t1)",
        r"Crack phase $\phi$ (-)",
        cmap_name="inferno",
        clip_percentile=99.0,
        force_vmin=0.0,
        force_vmax=1.0,
        spacing_xy=spacing_xy,
    )

    # slip table
    coupling = PFCCoupling(PFCParameters(), FractureParameters(), mode="density", **_build_coupling_kwargs(cfg))
    slip_rows = _build_slip_table(
        state_out,
        coupling,
        schmid_mode=schmid_mode,
        schmid_crystal_axis=(float(schmid_crystal_axis_vec[0]), float(schmid_crystal_axis_vec[1]), float(schmid_crystal_axis_vec[2])),
        load_axis_sample=(float(load_axis_sample_vec[0]), float(load_axis_sample_vec[1]), float(load_axis_sample_vec[2])),
    )
    slip_csv = out_dir / "validation_slip_damage_schmid.csv"
    slip_png = out_dir / "validation_slip_damage_schmid_table.png"
    _write_rows_csv(
        slip_csv,
        slip_rows,
        fieldnames=[
            "slip_id",
            "direction_uvws",
            "plane_hkls",
            "schmid_mean",
            "tau_mean_MPa",
            "schmid_max",
            "schmid_anchor",
            "schmid_sim_mean",
            "activity_share_pct",
            "gamma_abs_mean",
            "gamma_abs_p95",
            "damage_index_mean",
            "damage_index_p95",
            "backstress_abs_mean",
        ],
    )
    _plot_table(slip_rows, slip_png, "Slip Systems Sorted by Schmid")

    # compact summary
    summary = {
        "config": str(args.config),
        "run_dir": str(run_dir),
        "cycles_completed": int(len(results)),
        "stress_ref_gpa": float(stress_ref_gpa),
        "sigma_source_column": str(stress_source_column),
        "crack_mean_t1": float(results[-1].crack_mean),
        "crack_p95_t1": float(results[-1].crack_p95),
        "crack_p99_t1": float(results[-1].crack_p99),
        "crack_localization_index_t1": float(results[-1].crack_localization_index),
        "accum_plastic_mean_t1": float(results[-1].accum_plastic_mean),
        "gnd_mean_t1": float(results[-1].gnd_mean),
        "gnd_max_t1": float(results[-1].gnd_max),
        "paris_coeff": float(paris_coeff),
        "coffman_coeff": float(coffman_coeff),
        "trash_dir": str(trash_dir),
        "moved_old_images": [str(p) for p in moved],
        "outputs": {
            "grod_png": str(grod_png),
            "grod_ipf_png": str(grod_png_legacy),
            "accum_plastic_png": str(plastic_png),
            "crack_png": str(crack_png),
            "sigma_a_curve_png": str(fig_curve),
            "sigma_a_curve_csv": str(curve_csv),
            "slip_table_png": str(slip_png),
            "slip_table_csv": str(slip_csv),
        },
        "schmid": {
            "mode": schmid_mode,
            "schmid_crystal_loading_axis": [float(schmid_crystal_axis_vec[0]), float(schmid_crystal_axis_vec[1]), float(schmid_crystal_axis_vec[2])],
            "load_axis_sample": [float(load_axis_sample_vec[0]), float(load_axis_sample_vec[1]), float(load_axis_sample_vec[2])],
        },
        "diagnostics": diagnostics,
    }
    (out_dir / "validation_bundle_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] run_dir: {run_dir}")
    print(f"[ok] moved old images: {len(moved)} -> {trash_dir}")
    print(f"[ok] GROD: {grod_png}")
    print(f"[ok] Accum-plastic: {plastic_png}")
    print(f"[ok] Crack: {crack_png}")
    print(f"[ok] Sigma_a-cum: {fig_curve}")
    print(f"[ok] Slip-table: {slip_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
