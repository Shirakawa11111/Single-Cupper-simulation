"""
Generate v3-style poster panels from real simulation outputs (LAMMPS trajectory).

Outputs keep the same panel naming/layout convention as synthetic_poster_v3.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml  # type: ignore
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rw_lab.lammpstrj import Lammpstrj


def gbr_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("gbr", ["#00aa55", "#1f77ff", "#e41a1c"], N=256)


def _point_values_from_field3d(field: np.ndarray) -> np.ndarray:
    return field.transpose(2, 1, 0).reshape(-1)


def _point_values_from_field2d(field2d: np.ndarray, nz: int) -> np.ndarray:
    field3d = np.repeat(field2d[:, :, None], nz, axis=2)
    return _point_values_from_field3d(field3d)


def _load_cfg(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")
    vc = raw.get("virtual_cycle", raw)
    if not isinstance(vc, dict):
        raise ValueError("Config virtual_cycle must be a mapping.")
    return vc


def _read_frame(path: Path, shape: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    nx, ny, nz = shape
    n = nx * ny * nz
    trj = Lammpstrj.read(path)
    if len(trj.atoms) != n:
        raise ValueError(f"Atom count mismatch: {len(trj.atoms)} vs expected {n}")

    points = np.zeros((n, 3), dtype=float)
    fields: dict[str, np.ndarray] = {
        "crack": np.zeros(shape, dtype=float),
        "accum_plastic": np.zeros(shape, dtype=float),
        "psi": np.zeros(shape, dtype=float),
        "gnd_density": np.zeros(shape, dtype=float),
        "stress_vm": np.zeros(shape, dtype=float),
    }

    for atom_id, row in trj.atoms.items():
        idx = atom_id - 1
        i = idx % nx
        j = (idx // nx) % ny
        k = idx // (nx * ny)
        points[idx, 0] = row.get("x", float(i))
        points[idx, 1] = row.get("y", float(j))
        points[idx, 2] = row.get("z", float(k))
        fields["crack"][i, j, k] = row.get("crack", 0.0)
        fields["accum_plastic"][i, j, k] = row.get("accum_plastic", 0.0)
        fields["psi"][i, j, k] = row.get("psi", 0.0)
        fields["gnd_density"][i, j, k] = row.get("gnd_density", 0.0)
        fields["stress_vm"][i, j, k] = row.get("stress_vm", 0.0)

    return points, fields


def _surface_tri_notch_from_box(
    notch_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ny: int,
) -> dict[str, float]:
    (x0, x1), (_, _), (z0, z1) = notch_box
    y_top = float(ny - 1)
    depth = max(2.0, 0.22 * ny)
    y_tip = max(0.0, y_top - depth)
    return {
        "x_center": 0.5 * (x0 + x1),
        "x_half_width": 0.6 * (x1 - x0),
        "y_top": y_top,
        "y_tip": y_tip,
        "z0": z0,
        "z1": z1,
    }


def _surface_tri_mask_points(points: np.ndarray, tri: dict[str, float]) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    frac = np.clip((y - tri["y_tip"]) / max(tri["y_top"] - tri["y_tip"], 1e-9), 0.0, 1.0)
    half_w = tri["x_half_width"] * frac
    return (
        (y >= tri["y_tip"]) & (y <= tri["y_top"]) &
        (np.abs(x - tri["x_center"]) <= half_w) &
        (z >= tri["z0"]) & (z <= tri["z1"])
    )


def _surface_tri_mask_xy(nx: int, ny: int, tri: dict[str, float]) -> np.ndarray:
    x = np.arange(nx, dtype=float)[:, None]
    y = np.arange(ny, dtype=float)[None, :]
    frac = np.clip((y - tri["y_tip"]) / max(tri["y_top"] - tri["y_tip"], 1e-9), 0.0, 1.0)
    half_w = tri["x_half_width"] * frac
    return (y >= tri["y_tip"]) & (y <= tri["y_top"]) & (np.abs(x - tri["x_center"]) <= half_w)


def _add_surface_tri_notch(ax, tri: dict[str, float]) -> None:
    xc = tri["x_center"]
    w = tri["x_half_width"]
    y_tip = tri["y_tip"]
    y_top = tri["y_top"]
    z0, z1 = tri["z0"], tri["z1"]
    tri0 = [(xc - w, y_top, z0), (xc + w, y_top, z0), (xc, y_tip, z0)]
    tri1 = [(xc - w, y_top, z1), (xc + w, y_top, z1), (xc, y_tip, z1)]
    faces = [
        tri0,
        tri1,
        [tri0[0], tri0[1], tri1[1], tri1[0]],
        [tri0[1], tri0[2], tri1[2], tri1[1]],
        [tri0[2], tri0[0], tri1[0], tri1[2]],
    ]
    notch_mesh = Poly3DCollection(faces, facecolors=(1.0, 0.0, 0.0, 0.15), edgecolors="red", linewidths=1.0)
    ax.add_collection3d(notch_mesh)


def plot_atoms(
    points: np.ndarray,
    values: np.ndarray,
    title: str,
    out: Path,
    notch_tri: dict[str, float] | None = None,
    show_notch: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    cmap = gbr_cmap()
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    elif vmax - vmin < 1e-12:
        vmax = vmin + 1e-6
    norm = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    colors = cmap(norm)

    fig = plt.figure(figsize=(6.0, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=8, c=colors, alpha=0.9)

    if show_notch and notch_tri is not None:
        _add_surface_tri_notch(ax, notch_tri)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    try:
        xr = float(np.max(points[:, 0]) - np.min(points[:, 0]))
        yr = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        zr = float(np.max(points[:, 2]) - np.min(points[:, 2]))
        ax.set_box_aspect((max(xr, 1e-3), max(yr, 1e-3), max(zr, 1e-3)))
    except Exception:
        pass
    ax.set_xlim(float(np.min(points[:, 0])), float(np.max(points[:, 0])))
    ax.set_ylim(float(np.min(points[:, 1])), float(np.max(points[:, 1])))
    ax.set_zlim(float(np.min(points[:, 2])), float(np.max(points[:, 2])))

    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(values)
    plt.colorbar(mappable, ax=ax, shrink=0.65, pad=0.1)
    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_grod_map(grod_deg: np.ndarray, notch_mask: np.ndarray, out: Path, title: str, vmax_deg: float) -> None:
    cmap = gbr_cmap().copy()
    cmap.set_bad(color="white")
    grod_masked = grod_deg.copy()
    grod_masked[notch_mask] = np.nan
    plt.figure(figsize=(6.0, 4.0))
    im = plt.imshow(grod_masked.T, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax_deg, aspect="auto")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(im, label="GROD (°)")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_grod_pair_map(grod_t0: np.ndarray, grod_t1: np.ndarray, notch_mask: np.ndarray, out: Path, vmax_deg: float) -> None:
    cmap = gbr_cmap().copy()
    cmap.set_bad(color="white")
    t0 = grod_t0.copy()
    t1 = grod_t1.copy()
    t0[notch_mask] = np.nan
    t1[notch_mask] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    im0 = axes[0].imshow(t0.T, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax_deg, aspect="auto")
    im1 = axes[1].imshow(t1.T, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax_deg, aspect="auto")
    axes[0].set_title("GROD t0")
    axes[1].set_title("GROD t1")
    axes[0].set_xlabel("X")
    axes[1].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[1].set_ylabel("Y")
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9, label="GROD (°)")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _grod_from_proxy_fields(plastic_2d: np.ndarray, gnd_2d: np.ndarray, max_deg: float) -> tuple[np.ndarray, np.ndarray]:
    ref_ori = 0.55
    p_norm = plastic_2d / (float(np.max(plastic_2d)) + 1e-12)
    g_norm = gnd_2d / (float(np.max(gnd_2d)) + 1e-12)
    ori_t1 = np.clip(ref_ori + 0.18 * p_norm + 0.10 * g_norm, 0.0, 1.0)
    ori_delta = np.clip(np.abs(ori_t1 - ref_ori), 0.0, None)

    scale = float(np.percentile(ori_delta, 99.0))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.max(ori_delta))
    if not np.isfinite(scale) or scale <= 1e-12:
        grod_t1 = np.zeros_like(ori_delta)
    else:
        grod_t1 = np.clip(max_deg * ori_delta / scale, 0.0, max_deg)
    grod_t0 = np.zeros_like(grod_t1)
    return grod_t0, grod_t1


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


def _euler_from_proxy_fields(
    plastic_2d: np.ndarray,
    gnd_2d: np.ndarray,
    orientation_vector: list[float] | tuple[float, float, float],
    grod_max_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    phi1_0, Phi_0, phi2_0 = _base_euler_from_orientation_vector(orientation_vector)
    _, grod_t1 = _grod_from_proxy_fields(plastic_2d, gnd_2d, max_deg=grod_max_deg)
    p_norm = plastic_2d / (float(np.max(plastic_2d)) + 1e-12)
    g_norm = gnd_2d / (float(np.max(gnd_2d)) + 1e-12)
    gn = grod_t1 / max(grod_max_deg, 1e-12)

    euler_t0 = np.zeros((*plastic_2d.shape, 3), dtype=float)
    euler_t1 = np.zeros((*plastic_2d.shape, 3), dtype=float)
    euler_t0[..., 0] = phi1_0
    euler_t0[..., 1] = Phi_0
    euler_t0[..., 2] = phi2_0

    euler_t1[..., 0] = np.mod(phi1_0 + 16.0 * gn + 8.0 * (p_norm - 0.5), 360.0)
    euler_t1[..., 1] = np.clip(Phi_0 + 10.0 * gn + 6.0 * (g_norm - 0.5), 0.0, 180.0)
    euler_t1[..., 2] = np.mod(phi2_0 + 20.0 * gn + 10.0 * (p_norm - g_norm), 360.0)
    return euler_t0, euler_t1


def plot_euler_triplet_map(euler_t0: np.ndarray, euler_t1: np.ndarray, notch_mask: np.ndarray, out: Path) -> None:
    cmap = gbr_cmap().copy()
    cmap.set_bad(color="white")
    labels = ["phi1 (°)", "Phi (°)", "phi2 (°)"]
    vmaxs = [360.0, 180.0, 360.0]
    vmins = [0.0, 0.0, 0.0]
    t0 = euler_t0.copy()
    t1 = euler_t1.copy()
    for c in range(3):
        t0[..., c][notch_mask] = np.nan
        t1[..., c][notch_mask] = np.nan

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 6.8), constrained_layout=True)
    for c in range(3):
        im0 = axes[0, c].imshow(t0[..., c].T, origin="lower", cmap=cmap, vmin=vmins[c], vmax=vmaxs[c], aspect="auto")
        im1 = axes[1, c].imshow(t1[..., c].T, origin="lower", cmap=cmap, vmin=vmins[c], vmax=vmaxs[c], aspect="auto")
        axes[0, c].set_title(f"t0 {labels[c]}")
        axes[1, c].set_title(f"t1 {labels[c]}")
        axes[0, c].set_xlabel("X")
        axes[1, c].set_xlabel("X")
        axes[0, c].set_ylabel("Y")
        axes[1, c].set_ylabel("Y")
        fig.colorbar(im1, ax=[axes[0, c], axes[1, c]], shrink=0.88, label=labels[c])
        _ = im0
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v3-style poster panels from real simulation run outputs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("sim/configs/poster_real_v3_quick.yaml"),
        help="Config used for the run (for grid shape and notch metadata).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("sim/tests/runs/2026-02-12/poster_real_v3_quick"),
        help="Run directory containing lammpstrj outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/synthetic_poster_v3_real"),
        help="Output directory for v3-style panels.",
    )
    parser.add_argument("--grod-max-deg", type=float, default=15.0, help="Upper color scale for GROD map in degrees.")
    parser.add_argument("--notch-threshold", type=float, default=0.5, help="Kept for compatibility (unused in surface-notch mode).")
    args = parser.parse_args()

    vc_cfg = _load_cfg(args.config)
    shape = tuple(int(v) for v in vc_cfg.get("grid_shape", [32, 16, 8]))
    if len(shape) != 3:
        raise ValueError("grid_shape must have three integers.")
    notch_box_raw = vc_cfg.get("notch_box", None)
    notch_box = None
    notch_tri = None
    if isinstance(notch_box_raw, list) and len(notch_box_raw) == 3:
        notch_box = (
            (float(notch_box_raw[0][0]), float(notch_box_raw[0][1])),
            (float(notch_box_raw[1][0]), float(notch_box_raw[1][1])),
            (float(notch_box_raw[2][0]), float(notch_box_raw[2][1])),
        )
        notch_tri = _surface_tri_notch_from_box(notch_box, shape[1])

    lammpstrj_dir = args.run_dir / "lammpstrj"
    frames = sorted(lammpstrj_dir.glob("virtual_cycle_*.lammpstrj"))
    if len(frames) < 2:
        raise FileNotFoundError(f"Need at least two lammpstrj frames in {lammpstrj_dir}")
    t0_path, t1_path = frames[0], frames[-1]

    points0, fields0 = _read_frame(t0_path, shape)
    points1, fields1 = _read_frame(t1_path, shape)

    crack0_vals = _point_values_from_field3d(fields0["crack"])
    crack1_vals = _point_values_from_field3d(fields1["crack"])
    plastic1_vals = _point_values_from_field3d(fields1["accum_plastic"])
    gnd1_vals = _point_values_from_field3d(fields1["gnd_density"])

    if notch_tri is None:
        raise ValueError("notch_box is required for v3-style surface-notch visualization.")
    carve_mask = _surface_tri_mask_points(points0, notch_tri)
    keep_mask = ~carve_mask
    points_carved = points0[keep_mask]
    crack0_vals = crack0_vals[keep_mask]
    crack1_vals = crack1_vals[keep_mask]
    plastic1_vals = plastic1_vals[keep_mask]
    gnd1_vals = gnd1_vals[keep_mask]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    neutral = np.ones(len(points_carved)) * 0.5
    plot_atoms(
        points_carved,
        neutral,
        "Real run: single-crystal Cu + notch seed",
        out_dir / "panel_A_atoms.png",
        notch_tri=notch_tri,
        show_notch=True,
    )

    # Use fixed absolute scale to avoid t1 oversaturation.
    crack_vmax = max(1.0e-3, float(np.max(crack0_vals)), float(np.max(crack1_vals)))
    plot_atoms(
        points_carved,
        crack0_vals,
        "Crack phase-field (t0)",
        out_dir / "panel_B_crack_t0_atoms.png",
        vmin=0.0,
        vmax=crack_vmax,
    )
    plot_atoms(
        points_carved,
        crack1_vals,
        "Crack phase-field (t1)",
        out_dir / "panel_B_crack_t1_atoms.png",
        vmin=0.0,
        vmax=crack_vmax,
    )
    delta_vals = np.clip(crack1_vals - crack0_vals, 0.0, None)
    if float(np.max(delta_vals)) > 0.0:
        delta_plot = np.sqrt(delta_vals / float(np.max(delta_vals)))
    else:
        delta_plot = delta_vals.copy()
    plot_atoms(points_carved, delta_plot, "Crack growth Δφ (t1 - t0)", out_dir / "panel_B_crack_delta_atoms.png")

    try:
        import matplotlib.image as mpimg

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        img0 = mpimg.imread(out_dir / "panel_B_crack_t0_atoms.png")
        img1 = mpimg.imread(out_dir / "panel_B_crack_t1_atoms.png")
        imgd = mpimg.imread(out_dir / "panel_B_crack_delta_atoms.png")
        for ax, img, title in zip(axes, [img0, img1, imgd], ["t0", "t1", "Δφ"]):
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(title)
        fig.savefig(out_dir / "panel_B_crack_triplet.png", dpi=220)
        plt.close(fig)
    except Exception:
        pass

    plot_atoms(points_carved, plastic1_vals, "Accumulated plastic (t1)", out_dir / "panel_C_plastic_atoms.png")
    plot_atoms(points_carved, gnd1_vals, "GND density (t1)", out_dir / "panel_D_gnd_atoms.png")

    crack0_2d = np.max(fields0["crack"], axis=2)
    plastic1_2d = np.max(fields1["accum_plastic"], axis=2)
    gnd1_2d = np.max(fields1["gnd_density"], axis=2)

    grod_t0, grod_t1 = _grod_from_proxy_fields(plastic1_2d, gnd1_2d, max_deg=max(1e-6, args.grod_max_deg))
    grod_t1_vals = _point_values_from_field2d(grod_t1, shape[2])[keep_mask]
    plot_atoms(
        points_carved,
        grod_t1_vals,
        "GROD (t1, deg)",
        out_dir / "panel_E_grod_atoms.png",
        notch_tri=notch_tri,
        show_notch=True,
        vmin=0.0,
        vmax=max(1e-6, args.grod_max_deg),
    )
    notch_mask_2d = _surface_tri_mask_xy(shape[0], shape[1], notch_tri)
    vmax_deg = max(1e-6, args.grod_max_deg)
    plot_grod_map(grod_t0, notch_mask_2d, out_dir / "panel_E_grod_t0.png", "GROD (t0)", vmax_deg=vmax_deg)
    plot_grod_map(grod_t1, notch_mask_2d, out_dir / "panel_E_grod_t1.png", "GROD (t1, post-tension)", vmax_deg=vmax_deg)
    plot_grod_pair_map(grod_t0, grod_t1, notch_mask_2d, out_dir / "panel_E_grod_map.png", vmax_deg=vmax_deg)

    # Backward-compatible file names for existing poster assembly scripts.
    plot_grod_map(grod_t0, notch_mask_2d, out_dir / "panel_E_ebsd_t0.png", "GROD (t0)", vmax_deg=vmax_deg)
    plot_grod_map(grod_t1, notch_mask_2d, out_dir / "panel_E_ebsd_t1.png", "GROD (t1, post-tension)", vmax_deg=vmax_deg)
    plot_grod_pair_map(grod_t0, grod_t1, notch_mask_2d, out_dir / "panel_E_ebsd_map.png", vmax_deg=vmax_deg)

    orientation_vector = vc_cfg.get("orientation_vector", [1.0, 1.0, 1.0])
    if not isinstance(orientation_vector, list) or len(orientation_vector) != 3:
        orientation_vector = [1.0, 1.0, 1.0]
    euler_t0, euler_t1 = _euler_from_proxy_fields(
        plastic1_2d,
        gnd1_2d,
        orientation_vector=orientation_vector,
        grod_max_deg=vmax_deg,
    )
    plot_euler_triplet_map(euler_t0, euler_t1, notch_mask_2d, out_dir / "panel_F_euler_map.png")

    print(f"[ok] Real v3-style poster assets saved in {out_dir}")
    print(f"[info] t0={t0_path}")
    print(f"[info] t1={t1_path}")


if __name__ == "__main__":
    main()
