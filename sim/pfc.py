"""
Spectral Phase-Field Crystal evolver.

Defaults are non-dimensional and assume the mechanical solver provides
macro strain in the same (small-strain) measure. Optional hooks allow
adding extra chemical potential terms (e.g., stress-assisted PFC).
"""

from __future__ import annotations

from typing import Callable, Tuple
import numpy as np

from .energy import Array, PFCParameters
from .operators import GridSpec


class PFCEvolver:
    def __init__(
        self,
        grid: GridSpec,
        params: PFCParameters,
        dt: float = 1e-3,
        extra_mu: Callable[[Array], Array] | None = None,
        clip: float | None = None,
    ) -> None:
        self.grid = grid
        self.params = params
        self.dt = dt
        self.extra_mu = extra_mu  # optional external coupling term μ_extra(ψ)
        self.clip = clip  # optional soft guard; None disables clipping
        # 预先计算基础波矢量
        self.base_k_axes = self._compute_base_wave_numbers()
        self.k2 = self._compute_k2(strain=(0.0, 0.0, 0.0))

    def _compute_base_wave_numbers(self) -> list[Array]:
        axes = []
        for n, d in zip(self.grid.shape, self.grid.spacing):
            # 注意：这里计算的是 k，与 d 成反比
            k = 2 * np.pi * np.fft.fftfreq(n, d=d)
            axes.append(k)
        return axes

    def _compute_k2(self, strain: Tuple[float, float, float]) -> Array:
        """
        根据宏观应变修正波矢量。
        物理逻辑：
        拉伸 (eps > 0) -> 实际空间波长变大 -> 倒空间 k 应减小。
        采用坐标伸缩：k_grid = k_base / (1 + eps) ，确保拉伸时 k 变小，
        压缩时 k 变大。
        """
        k2 = np.zeros(self.grid.shape)
        for axis, k_base in enumerate(self.base_k_axes):
            eps = strain[axis]
            scale = 1.0 / (1.0 + eps)
            k_strained = k_base * scale
            
            shape = [1] * len(self.grid.shape)
            shape[axis] = -1
            k2 += k_strained.reshape(shape) ** 2
        return k2

    def update_strain(self, strain: Tuple[float, float, float]) -> None:
        self.k2 = self._compute_k2(strain)

    def chemical_potential(self, psi: Array) -> Array:
        psi_hat = np.fft.fftn(psi)
        # Swift-Hohenberg 算子
        operator = self.params.r + (self.params.q0**2 - self.k2) ** 2
        linear = np.fft.ifftn(operator * psi_hat).real
        nonlinear = self.params.u * psi**3
        mu = linear + nonlinear
        if self.extra_mu is not None:
            mu = mu + self.extra_mu(psi)
        return mu

    def step(self, psi: Array) -> Array:
        mu = self.chemical_potential(psi)
        mu_hat = np.fft.fftn(mu)
        psi_hat = np.fft.fftn(psi)
        # 守恒型动力学
        update = -self.k2 * mu_hat
        psi_new_hat = psi_hat + self.dt * update
        psi_new = np.fft.ifftn(psi_new_hat).real
        if self.clip is not None:
            psi_new = np.clip(psi_new, -self.clip, self.clip)
            psi_new = np.nan_to_num(psi_new, nan=0.0, posinf=self.clip, neginf=-self.clip)
        else:
            psi_new = np.nan_to_num(psi_new, nan=0.0)
        return psi_new


__all__ = ["PFCEvolver"]
