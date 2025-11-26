"""
Utilities for seeding discrete defects and converting them to continuous fields.

Usage:
    cfg = DefectConfig()  # or override attributes
    seeds = generate_defect_seeds(grid, cfg, rng=np.random.default_rng(0))
    fields = seeds_to_fields(grid, seeds, cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .operators import GridSpec


@dataclass
class DefectSeed:
    position: np.ndarray  # physical coordinates (x, y, z)
    kind: str
    direction: np.ndarray
    from_line: bool = False


@dataclass
class DefectConfig:
    """
    Parameters controlling defect seeding and field projection.

    seed_density: expected number of seeds per unit volume (same units as spacing).
    region_bounds: optional ((x0, x1), (y0, y1), (z0, z1)); defaults to full domain.
    type_probabilities: probability for each kind; keys: vacancy, interstitial, dislocation.
    orient_mode: "random" or "slip_system".
    max_seeds: cap to avoid exploding memory (defaults to 0.5 * voxel count).
    """

    seed_density: float = 1e14
    region_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] | None = None
    type_probabilities: Dict[str, float] = field(
        default_factory=lambda: {"vacancy": 0.34, "interstitial": 0.33, "dislocation": 0.33}
    )
    orient_mode: str = "random"  # or "slip_system"
    max_seeds: int | None = None

    sigma_normal: float = 1.0
    sigma_defect: float = 0.6
    weight_normal: float = 1.0
    weight_defect: float = 1.0
    vacancy_weight: float = -0.8
    interstitial_weight: float = 1.0
    dislocation_weight: float = 0.8

    crack_amplitude: float = 0.05
    plastic_amplitude: float = 0.02
    mean_target: float = 0.0
    peak_target: float = 0.25

    # Optional spatial weighting and line-defect support
    weight_field: np.ndarray | None = None  # same shape as grid; used as sampling weights
    mask_threshold: float = 0.0  # weights below this are treated as zero
    line_segments: List[dict] | None = None  # each: {"points": [(x,y,z), ...], "kind": "...", "direction": (optional)}
    line_step: float | None = None  # step for discretizing line segments; defaults to min spacing
    line_sigma_along: float = 0.2
    line_sigma_across: float = 0.05
    line_weight: float = 1.0


def _box_bounds_from_grid(grid: GridSpec) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    x = (0.0, (grid.shape[0] - 1) * grid.spacing[0])
    y = (0.0, (grid.shape[1] - 1) * grid.spacing[1])
    z = (0.0, (grid.shape[2] - 1) * grid.spacing[2])
    return x, y, z


def _normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(p, 0.0) for p in probs.values())
    if total <= 0:
        return {"vacancy": 1.0}
    return {k: max(v, 0.0) / total for k, v in probs.items()}


def _random_direction(rng: np.random.Generator) -> np.ndarray:
    vec = rng.standard_normal(3)
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def _sample_slip_direction(rng: np.random.Generator) -> np.ndarray:
    # FCC {111}<110> slip directions
    slip_dirs = np.array(
        [
            (1, 1, 0),
            (1, -1, 0),
            (1, 0, 1),
            (1, 0, -1),
            (0, 1, 1),
            (0, 1, -1),
        ],
        dtype=float,
    )
    idx = rng.integers(0, slip_dirs.shape[0])
    vec = slip_dirs[idx]
    return vec / (np.linalg.norm(vec) + 1e-12)


def generate_defect_seeds(grid: GridSpec, cfg: DefectConfig, rng: np.random.Generator | None = None) -> List[DefectSeed]:
    rng = np.random.default_rng() if rng is None else rng
    bounds = cfg.region_bounds or _box_bounds_from_grid(grid)
    (x0, x1), (y0, y1), (z0, z1) = bounds
    volume = max(x1 - x0, 0.0) * max(y1 - y0, 0.0) * max(z1 - z0, 0.0)
    n_target = int(np.ceil(cfg.seed_density * volume))
    max_allowed = cfg.max_seeds
    if max_allowed is None:
        max_allowed = int(0.5 * np.prod(grid.shape))
    n_target = max(1, min(n_target, max_allowed))

    # 1) deterministic line seeds (if any)
    seeds: List[DefectSeed] = []
    if cfg.line_segments:
        step = cfg.line_step or min(grid.spacing)
        for seg in cfg.line_segments:
            pts = np.asarray(seg.get("points", []), dtype=float)
            if pts.shape[0] < 2:
                continue
            kind = seg.get("kind", "dislocation")
            for i in range(len(pts) - 1):
                p0, p1 = pts[i], pts[i + 1]
                seg_vec = p1 - p0
                seg_len = np.linalg.norm(seg_vec)
                if seg_len < 1e-12:
                    continue
                n_sub = max(1, int(np.ceil(seg_len / step)))
                t = np.linspace(0.0, 1.0, n_sub, endpoint=True)
                dir_unit = seg_vec / seg_len
                direction = seg.get("direction")
                if direction is None:
                    direction = dir_unit
                for alpha in t:
                    pos = p0 + alpha * seg_vec
                    seeds.append(
                        DefectSeed(
                            position=pos,
                            kind=kind,
                            direction=np.asarray(direction, dtype=float),
                            from_line=True,
                        )
                    )

    remaining = max(0, n_target - len(seeds))
    if remaining == 0:
        return seeds

    probs = _normalize_probs(cfg.type_probabilities)
    kinds = list(probs.keys())
    weights = np.array([probs[k] for k in kinds], dtype=float)

    # 2) weighted or uniform sampling for the rest
    if cfg.weight_field is not None:
        w = np.array(cfg.weight_field, dtype=float)
        if w.shape != grid.shape:
            raise ValueError(f"weight_field shape {w.shape} != grid shape {grid.shape}")
        mask = w >= cfg.mask_threshold
        w_masked = np.where(mask, w, 0.0)
        # Apply region bounds mask
        coords_x = np.linspace(0, grid.spacing[0] * (grid.shape[0] - 1), grid.shape[0])
        coords_y = np.linspace(0, grid.spacing[1] * (grid.shape[1] - 1), grid.shape[1])
        coords_z = np.linspace(0, grid.spacing[2] * (grid.shape[2] - 1), grid.shape[2])
        X, Y, Z = np.meshgrid(coords_x, coords_y, coords_z, indexing="ij")
        region_mask = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1)
        w_masked = np.where(region_mask, w_masked, 0.0)
        flat = w_masked.ravel()
        total = flat.sum()
        if total > 0:
            p = flat / total
            idx = rng.choice(flat.size, size=remaining, replace=True, p=p)
            ix, iy, iz = np.unravel_index(idx, grid.shape)
            for a, b, c in zip(ix, iy, iz):
                pos = np.array([a * grid.spacing[0], b * grid.spacing[1], c * grid.spacing[2]])
                kind = rng.choice(kinds, p=weights)
                direction = _sample_slip_direction(rng) if (kind == "dislocation" and cfg.orient_mode == "slip_system") else _random_direction(rng)
                seeds.append(DefectSeed(position=pos, kind=kind, direction=direction))
            return seeds
        # fall back to uniform if weights zero

    for _ in range(remaining):
        pos = np.array(
            [
                rng.uniform(x0, x1),
                rng.uniform(y0, y1),
                rng.uniform(z0, z1),
            ]
        )
        kind = rng.choice(kinds, p=weights)
        if kind == "dislocation":
            direction = _sample_slip_direction(rng) if cfg.orient_mode == "slip_system" else _random_direction(rng)
        else:
            direction = _random_direction(rng)
        seeds.append(DefectSeed(position=pos, kind=kind, direction=direction))
    return seeds


def _apply_gaussian_3d(
    field: np.ndarray,
    center: np.ndarray,
    sigma: np.ndarray,
    weight: float,
    grid: GridSpec,
) -> None:
    dx, dy, dz = grid.spacing
    cx = center[0] / dx
    cy = center[1] / dy
    cz = center[2] / dz
    rad_x = int(np.ceil(3 * sigma[0] / dx))
    rad_y = int(np.ceil(3 * sigma[1] / dy))
    rad_z = int(np.ceil(3 * sigma[2] / dz))

    x_range = range(max(0, int(cx - rad_x)), min(grid.shape[0], int(cx + rad_x) + 1))
    y_range = range(max(0, int(cy - rad_y)), min(grid.shape[1], int(cy + rad_y) + 1))
    z_range = range(max(0, int(cz - rad_z)), min(grid.shape[2], int(cz + rad_z) + 1))

    if len(x_range) == 0 or len(y_range) == 0 or len(z_range) == 0:
        return

    sig2 = 2.0 * np.array(sigma) ** 2
    for ix in x_range:
        dx_val = ix * dx - center[0]
        for iy in y_range:
            dy_val = iy * dy - center[1]
            for iz in z_range:
                dz_val = iz * dz - center[2]
                expo = -(dx_val**2 / sig2[0] + dy_val**2 / sig2[1] + dz_val**2 / sig2[2])
                field[ix, iy, iz] += weight * np.exp(expo)


def seeds_to_fields(
    grid: GridSpec,
    seeds: Iterable[DefectSeed],
    cfg: DefectConfig,
) -> Dict[str, np.ndarray]:
    psi = np.zeros(grid.shape)
    crack = np.zeros_like(psi)
    plastic = np.zeros_like(psi)
    defect_mask = np.zeros_like(psi)
    line_mask = np.zeros_like(psi)

    sigma_default = np.array([cfg.sigma_normal, cfg.sigma_normal, cfg.sigma_normal], dtype=float)
    sigma_defect = np.array([cfg.sigma_defect, cfg.sigma_defect, cfg.sigma_defect], dtype=float)

    for seed in seeds:
        defect_mask_idx = (
            int(round(seed.position[0] / grid.spacing[0])),
            int(round(seed.position[1] / grid.spacing[1])),
            int(round(seed.position[2] / grid.spacing[2])),
        )
        if (
            0 <= defect_mask_idx[0] < grid.shape[0]
            and 0 <= defect_mask_idx[1] < grid.shape[1]
            and 0 <= defect_mask_idx[2] < grid.shape[2]
        ):
            defect_mask[defect_mask_idx] = 1.0
            if seed.from_line:
                line_mask[defect_mask_idx] = 1.0

        if seed.kind == "vacancy":
            sigma = sigma_defect
            weight = cfg.vacancy_weight
            crack_weight = cfg.crack_amplitude * 0.5
            plastic_weight = 0.0
        elif seed.kind == "interstitial":
            sigma = sigma_defect
            weight = cfg.interstitial_weight
            crack_weight = cfg.crack_amplitude * 0.3
            plastic_weight = cfg.plastic_amplitude * 0.5
        elif seed.kind == "dislocation":
            # Elongate along slip direction or provided line anisotropy
            dir_norm = seed.direction / (np.linalg.norm(seed.direction) + 1e-12)
            if seed.from_line:
                sigma = np.array(
                    [
                        cfg.line_sigma_along * abs(dir_norm[0]) + cfg.line_sigma_across,
                        cfg.line_sigma_along * abs(dir_norm[1]) + cfg.line_sigma_across,
                        cfg.line_sigma_along * abs(dir_norm[2]) + cfg.line_sigma_across,
                    ]
                )
                weight = cfg.line_weight
            else:
                anisotropy = np.abs(dir_norm) + 0.5
                sigma = sigma_defect * anisotropy
                weight = cfg.dislocation_weight
            crack_weight = cfg.crack_amplitude
            plastic_weight = cfg.plastic_amplitude
        else:
            sigma = sigma_default
            weight = cfg.weight_defect
            crack_weight = cfg.crack_amplitude * 0.2
            plastic_weight = cfg.plastic_amplitude * 0.2

        _apply_gaussian_3d(psi, seed.position, sigma, weight, grid)
        _apply_gaussian_3d(crack, seed.position, sigma, crack_weight, grid)
        _apply_gaussian_3d(plastic, seed.position, sigma, plastic_weight, grid)

    psi = _rescale_field(psi, cfg.mean_target, cfg.peak_target)
    crack = np.clip(crack, 0.0, 1.0)
    plastic = np.clip(plastic, 0.0, 1.0)
    out = {"psi": psi, "crack": crack, "plastic": plastic, "defect_mask": defect_mask}
    if line_mask.max() > 0:
        out["line_mask"] = line_mask
    return out


def _rescale_field(field: np.ndarray, mean_target: float, peak_target: float) -> np.ndarray:
    fld = field.copy()
    current_mean = float(fld.mean())
    fld -= current_mean
    current_peak = float(np.max(np.abs(fld))) + 1e-12
    fld *= peak_target / current_peak
    fld += mean_target
    return fld


__all__ = ["DefectConfig", "DefectSeed", "generate_defect_seeds", "seeds_to_fields"]
