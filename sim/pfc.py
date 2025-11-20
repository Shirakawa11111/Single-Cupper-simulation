"""
Spectral Phase-Field Crystal evolver.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from .energy import Array, PFCParameters
from .operators import GridSpec


class PFCEvolver:
    def __init__(self, grid: GridSpec, params: PFCParameters, dt: float = 1e-3) -> None:
        self.grid = grid
        self.params = params
        self.dt = dt
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
        拉伸 (eps > 0) -> 空间波长变大 -> 倒空间 k 应该变小。
        PFC 算子倾向于维持 q0 (比如 1.0)。
        我们需要变换坐标系，使得： k_phys = k_grid * (1 + eps)
        这样当 k_phys = q0 时，k_grid = q0 / (1 + eps) < q0 (实现了拉伸)。
        """
        k2 = np.zeros(self.grid.shape)
        for axis, k_base in enumerate(self.base_k_axes):
            eps = strain[axis]
            # 【修正点】除法改为乘法，实现正确的拉伸耦合
            k_strained = k_base * (1.0 + eps)
            
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
        return linear + nonlinear

    def step(self, psi: Array) -> Array:
        mu = self.chemical_potential(psi)
        mu_hat = np.fft.fftn(mu)
        psi_hat = np.fft.fftn(psi)
        # 守恒型动力学
        update = -self.k2 * mu_hat
        psi_new_hat = psi_hat + self.dt * update
        psi_new = np.fft.ifftn(psi_new_hat).real
        psi_new = np.clip(psi_new, -1.5, 1.5)
        return psi_new


__all__ = ["PFCEvolver"]