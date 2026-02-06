"""
Dislocation diagnostics: Nye tensor and GND density.
"""

from __future__ import annotations

import numpy as np

from .energy import Array
from .operators import FFTDifferentiator, GridSpec


def beta_p_from_slip(
    gamma_s: Array,
    slip_systems: list[tuple[Array, Array]],
    orientation: Array | None = None,
) -> Array:
    """
    Construct plastic distortion beta_p = sum_s gamma_s * (m ⊗ n).
    """
    if gamma_s.shape[0] != len(slip_systems):
        raise ValueError("gamma_s first dimension must match slip system count.")
    if orientation is not None:
        orientation = np.asarray(orientation)
        if orientation.shape not in ((3, 3), gamma_s.shape[1:] + (3, 3)):
            raise ValueError("orientation must be (3,3) or match gamma_s shape")

    beta_p = np.zeros(gamma_s.shape[1:] + (3, 3), dtype=float)
    for k, (m, n) in enumerate(slip_systems):
        if orientation is None:
            m_lab = m
            n_lab = n
            mn = np.outer(m_lab, n_lab)
        else:
            m_lab = np.einsum("...ij,j->...i", orientation, m, optimize=True)
            n_lab = np.einsum("...ij,j->...i", orientation, n, optimize=True)
            mn = np.einsum("...i,...j->...ij", m_lab, n_lab, optimize=True)
        beta_p += gamma_s[k][..., None, None] * mn
    return beta_p


def nye_tensor(beta_p: Array, grid: GridSpec) -> Array:
    """
    Compute Nye tensor alpha_ij = eps_jkl * d beta_p_il / dx_k.
    """
    fft = FFTDifferentiator(grid)
    alpha = np.zeros_like(beta_p)
    eps = np.zeros((3, 3, 3), dtype=float)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

    for i in range(3):
        for l in range(3):
            grads = fft.gradient(beta_p[..., i, l])
            for j in range(3):
                alpha[..., i, j] += (
                    eps[j, 0, l] * grads[0]
                    + eps[j, 1, l] * grads[1]
                    + eps[j, 2, l] * grads[2]
                )
    return alpha


def gnd_density(alpha: Array, burgers: float = 1.0) -> Array:
    """
    Compute GND density from Nye tensor (Frobenius norm).
    """
    burgers = max(float(burgers), 1e-12)
    sq = np.sum(alpha * alpha, axis=(-2, -1))
    sq = np.nan_to_num(sq, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
    return np.sqrt(np.clip(sq, 0.0, None)) / burgers


def gnd_from_slip(
    gamma_s: Array,
    slip_systems: list[tuple[Array, Array]],
    orientation: Array | None,
    grid: GridSpec,
    burgers: float = 1.0,
) -> tuple[Array, Array]:
    """
    Convenience wrapper: gamma_s -> beta_p -> Nye -> rho_GND.
    """
    beta_p = beta_p_from_slip(gamma_s, slip_systems, orientation=orientation)
    alpha = nye_tensor(beta_p, grid)
    rho = gnd_density(alpha, burgers=burgers)
    return rho, alpha


__all__ = ["beta_p_from_slip", "nye_tensor", "gnd_density", "gnd_from_slip"]
