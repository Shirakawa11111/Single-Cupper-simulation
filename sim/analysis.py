"""
Analysis utilities for fatigue metrics and crack length extraction.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .operators import GridSpec


def crack_length(
    phi: np.ndarray,
    grid: GridSpec,
    axis: int = 0,
    threshold: float = 0.95,
    x0: float = 0.0,
) -> float:
    """
    Estimate crack length along `axis` using a thresholded phase-field.
    Returns mean length across transverse lines, minus x0 (notch tip).
    """
    phi_axis = np.moveaxis(phi, axis, 0)
    n_axis = phi_axis.shape[0]
    idx_axis = np.arange(n_axis).reshape((n_axis,) + (1,) * (phi_axis.ndim - 1))
    hit = phi_axis >= threshold
    idx = np.where(hit, idx_axis, -1)
    max_idx = idx.max(axis=0)
    max_idx = np.maximum(max_idx, 0)
    length = max_idx * grid.spacing[axis] - x0
    length = np.maximum(length, 0.0)
    return float(np.mean(length))


def crack_growth_rate(a: Iterable[float], cycles: Iterable[float] | None = None) -> np.ndarray:
    """
    Compute da/dN from a crack length series.
    """
    a_arr = np.asarray(list(a), dtype=float)
    if cycles is None:
        return np.diff(a_arr)
    n_arr = np.asarray(list(cycles), dtype=float)
    return np.diff(a_arr) / np.diff(n_arr)


def cycle_range(values: Iterable[float], cycle_ids: Iterable[int]) -> dict[int, float]:
    """
    Compute per-cycle range for a scalar series (e.g., plastic proxy).
    """
    vals = np.asarray(list(values), dtype=float)
    cids = np.asarray(list(cycle_ids), dtype=int)
    out: dict[int, float] = {}
    for c in np.unique(cids):
        mask = cids == c
        if not np.any(mask):
            continue
        v = vals[mask]
        out[int(c)] = float(np.max(v) - np.min(v))
    return out


__all__ = ["crack_length", "crack_growth_rate", "cycle_range"]
