"""
Generate continuum phase-field style panels from real simulation outputs.

This script converts discrete atom-based trajectory fields into smooth 2D
continuum maps (projection over thickness) for poster-style presentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
import numpy as np
import yaml  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rw_lab.lammpstrj import Lammpstrj


def gbr_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("gbr", ["#00aa55", "#1f77ff", "#e41a1c"], N=256)


def _load_cfg(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")
    vc = raw.get("virtual_cycle", raw)
    if not isinstance(vc, dict):
        raise ValueError("Config virtual_cycle must be a mapping.")
    return vc


def _read_frame(path: Path, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    nx, ny, nz = shape
    n = nx * ny * nz
    trj = Lammpstrj.read(path)
    if len(trj.atoms) != n:
        raise ValueError(f"Atom count mismatch: {len(trj.atoms)} vs expected {n}")

    fields: dict[str, np.ndarray] = {
        "crack": np.zeros(shape, dtype=float),
        "accum_plastic": np.zeros(shape, dtype=float),
        "gnd_density": np.zeros(shape, dtype=float),
    }
    for atom_id, row in trj.atoms.items():
        idx = atom_id - 1
        i = idx % nx
        j = (idx // nx) % ny
        k = idx // (nx * ny)
        fields["crack"][i, j, k] = row.get("crack", 0.0)
        fields["accum_plastic"][i, j, k] = row.get("accum_plastic", 0.0)
        fields["gnd_density"][i, j, k] = row.get("gnd_density", 0.0)
    return fields


def _surface_tri_notch_from_box(
    notch_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ny: int,
) -> dict[str, float]:
    (x0, x1), (_, _), _ = notch_box
    y_top = float(ny - 1)
    depth = max(2.0, 0.22 * ny)
    y_tip = max(0.0, y_top - depth)
    return {
        "x_center": 0.5 * (x0 + x1),
        "x_half_width": 0.6 * (x1 - x0),
        "y_top": y_top,
        "y_tip": y_tip,
    }


def _surface_tri_mask_xy(nx: int, ny: int, tri: dict[str, float]) -> np.ndarray:
    x = np.arange(nx, dtype=float)[:, None]
    y = np.arange(ny, dtype=float)[None, :]
    frac = np.clip((y - tri["y_tip"]) / max(tri["y_top"] - tri["y_tip"], 1e-9), 0.0, 1.0)
    half_w = tri["x_half_width"] * frac
    return (y >= tri["y_tip"]) & (y <= tri["y_top"]) & (np.abs(x - tri["x_center"]) <= half_w)


def _smooth2d(arr: np.ndarray, passes: int = 2) -> np.ndarray:
    out = arr.astype(float, copy=True)
    for _ in range(max(0, passes)):
        p = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:] +
            2 * p[1:-1, :-2] + 4 * p[1:-1, 1:-1] + 2 * p[1:-1, 2:] +
            p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
        ) / 16.0
    return out


def _robust_limits(arr: np.ndarray, qlo: float = 2.0, qhi: float = 98.0) -> tuple[float, float]:
    vmin = float(np.percentile(arr, qlo))
    vmax = float(np.percentile(arr, qhi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _add_notch_outline(ax: plt.Axes, tri: dict[str, float], color: str = "white", lw: float = 1.8) -> None:
    x0 = tri["x_center"] - tri["x_half_width"]
    x1 = tri["x_center"] + tri["x_half_width"]
    y_top = tri["y_top"]
    y_tip = tri["y_tip"]
    poly = Polygon([(x0, y_top), (x1, y_top), (tri["x_center"], y_tip)], closed=True, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(poly)


def _plot_map(
    data: np.ndarray,
    title: str,
    out: Path,
    cmap,
    notch_tri: dict[str, float] | None = None,
    notch_mask: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
) -> None:
    arr = data.copy()
    if notch_mask is not None:
        arr = arr.copy()
        arr[notch_mask] = np.nan
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(arr.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bicubic", aspect="auto")
    if notch_tri is not None:
        _add_notch_outline(ax, notch_tri, color="white")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = plt.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_map_log10(
    data: np.ndarray,
    title: str,
    out: Path,
    cmap,
    notch_tri: dict[str, float] | None = None,
    notch_mask: np.ndarray | None = None,
    cbar_label: str = "log10(value)",
) -> None:
    arr = np.clip(data.astype(float, copy=True), 0.0, None)
    if notch_mask is not None:
        arr = arr.copy()
        arr[notch_mask] = np.nan

    valid = arr[np.isfinite(arr) & (arr > 0.0)]
    if valid.size == 0:
        # Fallback to avoid invalid log when field is all zeros.
        arr_log = np.zeros_like(arr)
        vmin, vmax = 0.0, 1.0
    else:
        floor = float(np.percentile(valid, 2.0))
        pos_min = float(np.min(valid))
        eps = max(1e-16, min(floor, pos_min))
        arr_log = np.full_like(arr, np.nan)
        finite = np.isfinite(arr)
        arr_log[finite] = np.log10(np.clip(arr[finite], eps, None))
        vmin, vmax = _robust_limits(arr_log[np.isfinite(arr_log)], 2.0, 98.0)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(arr_log.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bicubic", aspect="auto")
    if notch_tri is not None:
        _add_notch_outline(ax, notch_tri, color="white")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_panel_a(nx: int, ny: int, notch_tri: dict[str, float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.add_patch(Polygon([(0, 0), (nx - 1, 0), (nx - 1, ny - 1), (0, ny - 1)], closed=True, fill=False, edgecolor="#888", linewidth=1.5))
    notch_fill = Polygon(
        [
            (notch_tri["x_center"] - notch_tri["x_half_width"], notch_tri["y_top"]),
            (notch_tri["x_center"] + notch_tri["x_half_width"], notch_tri["y_top"]),
            (notch_tri["x_center"], notch_tri["y_tip"]),
        ],
        closed=True,
        facecolor="#d7301f",
        edgecolor="#a50f15",
        alpha=0.85,
        linewidth=1.2,
    )
    ax.add_patch(notch_fill)
    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ny - 1)
    ax.set_aspect("equal")
    ax.set_title("Continuum specimen + surface triangular notch")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_triplet(img_paths: list[Path], titles: list[str], out: Path) -> None:
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, p, t in zip(axes, img_paths, titles):
        ax.imshow(mpimg.imread(p))
        ax.set_axis_off()
        ax.set_title(t)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _grod_from_proxy_fields(plastic_2d: np.ndarray, gnd_2d: np.ndarray, max_deg: float) -> tuple[np.ndarray, np.ndarray]:
    ref_ori = 0.55
    p_norm = np.nan_to_num(plastic_2d, nan=0.0) / (float(np.nanmax(plastic_2d)) + 1e-12)
    g_norm = np.nan_to_num(gnd_2d, nan=0.0) / (float(np.nanmax(gnd_2d)) + 1e-12)
    ori_t1 = np.clip(ref_ori + 0.16 * p_norm + 0.08 * g_norm, 0.0, 1.0)
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
    p_norm = np.nan_to_num(plastic_2d, nan=0.0) / (float(np.nanmax(plastic_2d)) + 1e-12)
    g_norm = np.nan_to_num(gnd_2d, nan=0.0) / (float(np.nanmax(gnd_2d)) + 1e-12)
    gn = grod_t1 / max(grod_max_deg, 1e-12)

    euler_t0 = np.zeros((*plastic_2d.shape, 3), dtype=float)
    euler_t1 = np.zeros((*plastic_2d.shape, 3), dtype=float)
    euler_t0[..., 0] = phi1_0
    euler_t0[..., 1] = Phi_0
    euler_t0[..., 2] = phi2_0

    euler_t1[..., 0] = np.mod(phi1_0 + 14.0 * gn + 7.0 * (p_norm - 0.5), 360.0)
    euler_t1[..., 1] = np.clip(Phi_0 + 9.0 * gn + 5.0 * (g_norm - 0.5), 0.0, 180.0)
    euler_t1[..., 2] = np.mod(phi2_0 + 18.0 * gn + 9.0 * (p_norm - g_norm), 360.0)
    return euler_t0, euler_t1


def _plot_euler_triplet_map(
    euler_t0: np.ndarray,
    euler_t1: np.ndarray,
    notch_tri: dict[str, float],
    notch_mask: np.ndarray,
    out: Path,
) -> None:
    cmap = gbr_cmap().copy()
    cmap.set_bad(color="white")
    labels = ["phi1 (°)", "Phi (°)", "phi2 (°)"]
    vmins = [0.0, 0.0, 0.0]
    vmaxs = [360.0, 180.0, 360.0]
    t0 = euler_t0.copy()
    t1 = euler_t1.copy()
    for c in range(3):
        t0[..., c][notch_mask] = np.nan
        t1[..., c][notch_mask] = np.nan

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 6.8), constrained_layout=True)
    for c in range(3):
        im0 = axes[0, c].imshow(t0[..., c].T, origin="lower", cmap=cmap, vmin=vmins[c], vmax=vmaxs[c], interpolation="bicubic", aspect="auto")
        im1 = axes[1, c].imshow(t1[..., c].T, origin="lower", cmap=cmap, vmin=vmins[c], vmax=vmaxs[c], interpolation="bicubic", aspect="auto")
        _add_notch_outline(axes[0, c], notch_tri, color="white")
        _add_notch_outline(axes[1, c], notch_tri, color="white")
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
    parser = argparse.ArgumentParser(description="Generate continuum phase-field poster panels from real run outputs.")
    parser.add_argument("--config", type=Path, default=Path("sim/configs/poster_real_v3_surface_visible.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("sim/tests/runs/2026-02-12/poster_real_v3_surface_visible"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/synthetic_poster_v3_real_continuum"))
    parser.add_argument("--smooth-passes", type=int, default=2)
    parser.add_argument("--grod-max-deg", type=float, default=15.0)
    args = parser.parse_args()

    vc_cfg = _load_cfg(args.config)
    shape = tuple(int(v) for v in vc_cfg.get("grid_shape", [32, 16, 8]))
    if len(shape) != 3:
        raise ValueError("grid_shape must have 3 ints")
    nx, ny, nz = shape

    notch_box_raw = vc_cfg.get("notch_box", None)
    if not (isinstance(notch_box_raw, list) and len(notch_box_raw) == 3):
        raise ValueError("notch_box is required for continuum notch rendering.")
    notch_box = (
        (float(notch_box_raw[0][0]), float(notch_box_raw[0][1])),
        (float(notch_box_raw[1][0]), float(notch_box_raw[1][1])),
        (float(notch_box_raw[2][0]), float(notch_box_raw[2][1])),
    )
    notch_tri = _surface_tri_notch_from_box(notch_box, ny)
    notch_mask_xy = _surface_tri_mask_xy(nx, ny, notch_tri)

    lammpstrj_dir = args.run_dir / "lammpstrj"
    frames = sorted(lammpstrj_dir.glob("virtual_cycle_*.lammpstrj"))
    if len(frames) < 2:
        raise FileNotFoundError(f"Need >=2 frames in {lammpstrj_dir}")
    f0 = _read_frame(frames[0], shape)
    f1 = _read_frame(frames[-1], shape)

    # thickness-projected continuum maps
    crack0 = np.max(f0["crack"], axis=2)
    crack1 = np.max(f1["crack"], axis=2)
    plastic1 = np.max(f1["accum_plastic"], axis=2)
    gnd1 = np.max(f1["gnd_density"], axis=2)

    crack0 = _smooth2d(crack0, passes=args.smooth_passes)
    crack1 = _smooth2d(crack1, passes=args.smooth_passes)
    plastic1 = _smooth2d(plastic1, passes=args.smooth_passes)
    gnd1 = _smooth2d(gnd1, passes=args.smooth_passes)

    # enforce notch cut-out at surface
    crack0[notch_mask_xy] = np.nan
    crack1[notch_mask_xy] = np.nan
    plastic1[notch_mask_xy] = np.nan
    gnd1[notch_mask_xy] = np.nan

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = gbr_cmap()

    _plot_panel_a(nx, ny, notch_tri, out_dir / "panel_A_continuum_notch.png")

    # Use per-panel robust scaling to avoid "full red t1".
    vmin0, vmax0 = _robust_limits(np.nan_to_num(crack0, nan=np.nanmedian(crack0)), 2.0, 98.0)
    vmin1, vmax1 = _robust_limits(np.nan_to_num(crack1, nan=np.nanmedian(crack1)), 2.0, 98.0)
    _plot_map(crack0, "Crack phase-field (t0, continuum)", out_dir / "panel_B_crack_t0_continuum.png", cmap, notch_tri, notch_mask_xy, vmin0, vmax0, cbar_label="phi")
    _plot_map(crack1, "Crack phase-field (t1, continuum)", out_dir / "panel_B_crack_t1_continuum.png", cmap, notch_tri, notch_mask_xy, vmin1, vmax1, cbar_label="phi")

    delta = np.clip(np.nan_to_num(crack1 - crack0, nan=0.0), 0.0, None)
    valid_delta = delta[~notch_mask_xy]
    if valid_delta.size > 0 and float(np.max(valid_delta)) > 0:
        d_lo = float(np.percentile(valid_delta, 5.0))
        d_hi = float(np.percentile(valid_delta, 99.0))
        if d_hi <= d_lo:
            d_lo = float(np.min(valid_delta))
            d_hi = float(np.max(valid_delta))
        if d_hi <= d_lo:
            delta_show = np.zeros_like(delta)
        else:
            delta_show = np.clip((delta - d_lo) / (d_hi - d_lo), 0.0, 1.0)
            delta_show = np.sqrt(delta_show)
    else:
        delta_show = np.zeros_like(delta)
    _plot_map(delta_show, "Crack growth Delta phi (t1 - t0, continuum)", out_dir / "panel_B_crack_delta_continuum.png", cmap, notch_tri, notch_mask_xy, 0.0, 1.0, cbar_label="normalized Delta phi")

    _plot_triplet(
        [
            out_dir / "panel_B_crack_t0_continuum.png",
            out_dir / "panel_B_crack_t1_continuum.png",
            out_dir / "panel_B_crack_delta_continuum.png",
        ],
        ["t0", "t1", "Delta phi"],
        out_dir / "panel_B_crack_triplet_continuum.png",
    )

    pmin, pmax = _robust_limits(np.nan_to_num(plastic1, nan=0.0), 2.0, 98.0)
    _plot_map(plastic1, "Accumulated plastic (t1, continuum)", out_dir / "panel_C_plastic_continuum.png", cmap, notch_tri, notch_mask_xy, pmin, pmax, cbar_label="accum_plastic")

    gmin, gmax = _robust_limits(np.nan_to_num(gnd1, nan=0.0), 2.0, 98.0)
    _plot_map(gnd1, "GND density (t1, continuum)", out_dir / "panel_D_gnd_continuum.png", cmap, notch_tri, notch_mask_xy, gmin, gmax, cbar_label="gnd_density")
    _plot_map_log10(
        gnd1,
        "GND density (t1, continuum, log10)",
        out_dir / "panel_D_gnd_continuum_log.png",
        cmap,
        notch_tri,
        notch_mask_xy,
        cbar_label="log10(gnd_density)",
    )

    vmax_deg = max(1e-6, args.grod_max_deg)
    grod_t0, grod_t1 = _grod_from_proxy_fields(plastic1, gnd1, max_deg=vmax_deg)
    grod_t0[notch_mask_xy] = np.nan
    grod_t1[notch_mask_xy] = np.nan
    _plot_map(grod_t0, "GROD (t0, continuum)", out_dir / "panel_E_grod_t0_continuum.png", cmap, notch_tri, notch_mask_xy, 0.0, vmax_deg, cbar_label="GROD (°)")
    _plot_map(grod_t1, "GROD (t1, continuum)", out_dir / "panel_E_grod_t1_continuum.png", cmap, notch_tri, notch_mask_xy, 0.0, vmax_deg, cbar_label="GROD (°)")
    # Backward-compatible names.
    _plot_map(grod_t0, "GROD (t0, continuum)", out_dir / "panel_E_ebsd_t0_continuum.png", cmap, notch_tri, notch_mask_xy, 0.0, vmax_deg, cbar_label="GROD (°)")
    _plot_map(grod_t1, "GROD (t1, continuum)", out_dir / "panel_E_ebsd_t1_continuum.png", cmap, notch_tri, notch_mask_xy, 0.0, vmax_deg, cbar_label="GROD (°)")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    im0 = axes[0].imshow(grod_t0.T, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax_deg, interpolation="bicubic", aspect="auto")
    im1 = axes[1].imshow(grod_t1.T, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax_deg, interpolation="bicubic", aspect="auto")
    _add_notch_outline(axes[0], notch_tri, color="white")
    _add_notch_outline(axes[1], notch_tri, color="white")
    axes[0].set_title("GROD t0 (continuum)")
    axes[1].set_title("GROD t1 (continuum)")
    axes[0].set_xlabel("X")
    axes[1].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[1].set_ylabel("Y")
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9, label="GROD (°)")
    fig.savefig(out_dir / "panel_E_grod_map_continuum.png", dpi=220)
    fig.savefig(out_dir / "panel_E_ebsd_map_continuum.png", dpi=220)
    plt.close(fig)

    orientation_vector = vc_cfg.get("orientation_vector", [1.0, 1.0, 1.0])
    if not isinstance(orientation_vector, list) or len(orientation_vector) != 3:
        orientation_vector = [1.0, 1.0, 1.0]
    euler_t0, euler_t1 = _euler_from_proxy_fields(
        plastic1,
        gnd1,
        orientation_vector=orientation_vector,
        grod_max_deg=vmax_deg,
    )
    _plot_euler_triplet_map(
        euler_t0,
        euler_t1,
        notch_tri=notch_tri,
        notch_mask=notch_mask_xy,
        out=out_dir / "panel_F_euler_map_continuum.png",
    )

    print(f"[ok] Continuum v3-style assets saved in {out_dir}")
    print(f"[info] t0={frames[0]}")
    print(f"[info] t1={frames[-1]}")


if __name__ == "__main__":
    main()
