"""
Generate 3D atomic-lattice style panels for all fields (synthetic).
- All panels rendered as 3D FCC lattice point clouds
- Colors mapped with Green -> Blue -> Red

NOTE: This is illustrative (synthetic), not experimental truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap


def sigmoid(x: np.ndarray, k: float = 10.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * x))


def make_fcc_points(nx: int, ny: int, nz: int, a: float = 1.0) -> np.ndarray:
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )
    pts = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell = np.array([i, j, k], dtype=float)
                for b in basis:
                    pts.append((cell + b) * a)
    return np.array(pts)


def notch_params_atoms() -> dict[str, float]:
    return {
        "x0": 10.0,
        "x1": 14.0,
        "z0": 2.0,
        "z1": 4.0,
        "y_top": 7.8,
        "y_tip": 6.0,
    }


def triangular_notch_mask_points(points: np.ndarray, notch: dict[str, float]) -> np.ndarray:
    x0, x1 = notch["x0"], notch["x1"]
    z0, z1 = notch["z0"], notch["z1"]
    y_top, y_tip = notch["y_top"], notch["y_tip"]
    zc = 0.5 * (z0 + z1)
    half_w = 0.5 * (z1 - z0)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    rel = 1.0 - np.abs(z - zc) / max(half_w, 1e-9)
    rel = np.clip(rel, 0.0, 1.0)
    y_cut = y_top - rel * (y_top - y_tip)
    return (
        (x >= x0) & (x <= x1) &
        (z >= z0) & (z <= z1) &
        (y >= y_cut) & (y <= y_top)
    )


def carve_notch(points: np.ndarray, notch: dict[str, float]) -> np.ndarray:
    return points[~triangular_notch_mask_points(points, notch)]


def notch_params_2d(nx: int, ny: int) -> dict[str, float]:
    return {
        "x_center": 0.5 * nx,
        "x_half_width": nx * 0.08,
        "y_tip": ny * 0.82,
        "y_top": ny - 1.0,
    }


def triangular_notch_mask_2d(X: np.ndarray, Y: np.ndarray, notch: dict[str, float]) -> np.ndarray:
    x_center = notch["x_center"]
    x_half_width = notch["x_half_width"]
    y_tip = notch["y_tip"]
    y_top = notch["y_top"]

    frac = np.clip((Y - y_tip) / max(y_top - y_tip, 1e-9), 0.0, 1.0)
    half_w_y = x_half_width * frac
    return (
        (Y >= y_tip) & (Y <= y_top) &
        (np.abs(X - x_center) <= half_w_y)
    )


def synthetic_fields(nx: int, ny: int, t: float) -> dict[str, np.ndarray]:
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    notch = notch_params_2d(nx, ny)
    notch_mask = triangular_notch_mask_2d(X, Y, notch)

    # crack front (downward growth) - local and short
    y_front = notch["y_tip"] - t * (ny * 0.18)
    x0 = notch["x_center"]
    wx = nx * 0.03
    length = ny * (0.03 + 0.12 * t)
    band = sigmoid((y_front - Y) / (ny * 0.02), k=8.0) * sigmoid((Y - (y_front - length)) / (ny * 0.02), k=8.0)
    core = np.exp(-((X - x0) / wx) ** 2)
    crack = core * band

    crack[notch_mask] = 1.0
    crack = np.clip(crack, 0.0, 1.0)

    # plastic field near crack tip
    tip = np.array([notch["x_center"], y_front])
    r2 = (X - tip[0]) ** 2 + (Y - tip[1]) ** 2
    plastic = 0.6 * np.exp(-r2 / (2.0 * (nx * 0.08) ** 2)) * (0.4 + 0.6 * crack)

    # GND proxy = |grad(plastic)|
    gx, gy = np.gradient(plastic)
    gnd = np.sqrt(gx * gx + gy * gy)

    return {
        "crack": crack,
        "plastic": plastic,
        "gnd": gnd,
        "notch": notch,
    }


def gbr_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("gbr", ["#00aa55", "#1f77ff", "#e41a1c"], N=256)


def sample_field_on_points(points: np.ndarray, field2d: np.ndarray, x_max: float, y_max: float) -> np.ndarray:
    ix = np.clip((points[:, 0] / x_max) * (field2d.shape[0] - 1), 0, field2d.shape[0] - 1)
    iy = np.clip((points[:, 1] / y_max) * (field2d.shape[1] - 1), 0, field2d.shape[1] - 1)
    ix = ix.astype(int)
    iy = iy.astype(int)
    return field2d[ix, iy]


def add_notch_prism(ax, notch: dict[str, float]) -> None:
    x0, x1 = notch["x0"], notch["x1"]
    z0, z1 = notch["z0"], notch["z1"]
    y_top, y_tip = notch["y_top"], notch["y_tip"]
    zc = 0.5 * (z0 + z1)
    tri0 = [(x0, y_top, z0), (x0, y_top, z1), (x0, y_tip, zc)]
    tri1 = [(x1, y_top, z0), (x1, y_top, z1), (x1, y_tip, zc)]
    faces = [
        tri0,
        tri1,
        [tri0[0], tri0[1], tri1[1], tri1[0]],
        [tri0[1], tri0[2], tri1[2], tri1[1]],
        [tri0[2], tri0[0], tri1[0], tri1[2]],
    ]
    mesh = Poly3DCollection(
        faces,
        facecolors=(1.0, 0.0, 0.0, 0.10),
        edgecolors="red",
        linewidths=1.0,
    )
    ax.add_collection3d(mesh)


def plot_atoms(points: np.ndarray, values: np.ndarray, title: str, out: Path, notch=None, show_notch=False) -> None:
    cmap = gbr_cmap()
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    elif vmax - vmin < 1e-12:
        vmax = vmin + 1e-6
    norm = (values - vmin) / (vmax - vmin)
    colors = cmap(norm)

    fig = plt.figure(figsize=(6.0, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=8, c=colors, alpha=0.9)

    if show_notch and notch is not None:
        add_notch_prism(ax, notch)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    try:
        ax.set_box_aspect((24, 8, 6))
    except Exception:
        pass
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 8)
    ax.set_zlim(0, 6)

    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(values)
    plt.colorbar(mappable, ax=ax, shrink=0.65, pad=0.1)

    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)





def make_ebsd_t0(shape: tuple[int, int], base: float = 0.55) -> np.ndarray:
    return np.ones(shape) * base


def make_ebsd_t1(plastic: np.ndarray, gnd: np.ndarray, notch: dict[str, float], base: float = 0.55) -> np.ndarray:
    plastic = np.nan_to_num(plastic)
    gnd = np.nan_to_num(gnd)
    p_norm = plastic / (float(plastic.max()) + 1e-12)
    g_norm = gnd / (float(gnd.max()) + 1e-12)
    ori = base + 0.18 * p_norm + 0.08 * g_norm

    nx, ny = plastic.shape
    x = np.linspace(0, nx - 1, nx)[:, None]
    y = np.linspace(0, ny - 1, ny)[None, :]
    x0 = notch["x_center"]
    y0 = notch["y_tip"] - 0.10 * ny
    long_range = np.exp(-((x - x0) / (0.09 * nx)) ** 2) * np.exp(-((y - y0) / (0.20 * ny)) ** 2)
    ori += 0.07 * long_range
    return np.clip(ori, 0.0, 1.0)


def plot_ebsd_map(ori: np.ndarray, notch: dict[str, float], out: Path, title: str) -> None:
    cmap = gbr_cmap()
    cmap = cmap.copy()
    cmap.set_bad(color="white")
    ori_masked = ori.copy()
    nx, ny = ori.shape
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    notch_mask = triangular_notch_mask_2d(X, Y, notch)
    ori_masked[notch_mask] = np.nan  # surface notch cut-out (no data)
    plt.figure(figsize=(6.0, 4.0))
    im = plt.imshow(ori_masked.T, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(im, label="orientation")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_ebsd_pair_map(ori_t0: np.ndarray, ori_t1: np.ndarray, notch: dict[str, float], out: Path) -> None:
    cmap = gbr_cmap()
    cmap = cmap.copy()
    cmap.set_bad(color="white")

    nx, ny = ori_t0.shape
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    notch_mask = triangular_notch_mask_2d(X, Y, notch)

    t0 = ori_t0.copy()
    t1 = ori_t1.copy()
    t0[notch_mask] = np.nan
    t1[notch_mask] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    im0 = axes[0].imshow(t0.T, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    im1 = axes[1].imshow(t1.T, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    axes[0].set_title("EBSD t0")
    axes[1].set_title("EBSD t1")
    axes[0].set_xlabel("X")
    axes[1].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[1].set_ylabel("Y")
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9, label="orientation")
    fig.savefig(out, dpi=220)
    plt.close(fig)

def main() -> None:
    out_dir = Path("docs/synthetic_poster_v3")
    out_dir.mkdir(parents=True, exist_ok=True)

    # lattice (rectangular)
    pts = make_fcc_points(24, 8, 6, a=1.0)
    notch_atoms = notch_params_atoms()
    pts_carved = carve_notch(pts, notch_atoms)

    # fields (2D), sampled onto lattice
    f0 = synthetic_fields(160, 80, t=0.0)
    f1 = synthetic_fields(160, 80, t=1.0)

    x_max, y_max = 24.0, 8.0

    crack0_vals = sample_field_on_points(pts_carved, f0["crack"], x_max, y_max)
    crack1_vals = sample_field_on_points(pts_carved, f1["crack"], x_max, y_max)
    plastic_vals = sample_field_on_points(pts_carved, f1["plastic"], x_max, y_max)
    gnd_vals = sample_field_on_points(pts_carved, f1["gnd"], x_max, y_max)

    # EBSD-like maps based on deformation (t0 vs t1)
    ebsd_t0 = make_ebsd_t0(f0["plastic"].shape, base=0.55)
    ebsd_t1 = make_ebsd_t1(f1["plastic"], f1["gnd"], f1["notch"], base=0.55)
    ebsd_t0_vals = sample_field_on_points(pts_carved, ebsd_t0, x_max, y_max)

    # Panel A: lattice with red notch box (keep from v2)
    # Use constant values for neutral color
    neutral = np.ones(len(pts_carved)) * 0.5
    plot_atoms(
        pts_carved,
        neutral,
        "Single-crystal Cu (FCC) + triangular surface notch",
        out_dir / "panel_A_atoms.png",
        notch=notch_atoms,
        show_notch=True,
    )

    # Panel B: crack t0 / t1
    delta_vals = np.clip(crack1_vals - crack0_vals, 0.0, None)
    # amplify low contrast for visibility
    delta_vals = np.sqrt(delta_vals / (delta_vals.max() + 1e-12))

    plot_atoms(pts_carved, crack0_vals, "Crack phase-field (t0)", out_dir / "panel_B_crack_t0_atoms.png")
    plot_atoms(pts_carved, crack1_vals, "Crack phase-field (t1)", out_dir / "panel_B_crack_t1_atoms.png")

    plot_atoms(pts_carved, delta_vals, "Crack growth Δφ (t1 - t0)", out_dir / "panel_B_crack_delta_atoms.png")

    # Composite B: t0 / t1 / delta in one row (for readability)
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

    # Panel C/D/E: plastic, GND, EBSD
    plot_atoms(pts_carved, plastic_vals, "Plastic field (t1)", out_dir / "panel_C_plastic_atoms.png")
    plot_atoms(pts_carved, gnd_vals, "GND proxy (t1)", out_dir / "panel_D_gnd_atoms.png")
    plot_atoms(
        pts_carved,
        ebsd_t0_vals,
        "EBSD orientation (t0, triangular notch)",
        out_dir / "panel_E_ebsd_atoms.png",
        notch=notch_atoms,
        show_notch=True,
    )
    plot_ebsd_map(ebsd_t0, f0["notch"], out_dir / "panel_E_ebsd_t0.png", "EBSD (t0, single crystal, triangular notch)")
    plot_ebsd_map(ebsd_t1, f1["notch"], out_dir / "panel_E_ebsd_t1.png", "EBSD (t1, post-tension local rotation)")
    plot_ebsd_pair_map(ebsd_t0, ebsd_t1, f1["notch"], out_dir / "panel_E_ebsd_map.png")

    print(f"[ok] Synthetic poster v3 assets saved in {out_dir}")


if __name__ == "__main__":
    main()
