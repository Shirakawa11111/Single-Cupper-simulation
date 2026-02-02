"""
Spectral Phase-Field Crystal evolver.

Defaults are non-dimensional and assume the mechanical solver provides
macro strain in the same (small-strain) measure. Optional hooks allow
adding extra chemical potential terms (e.g., stress-assisted PFC).
"""

from __future__ import annotations

import os
from typing import Callable, Literal, Tuple
import numpy as np

from .energy import Array, PFCParameters
from .operators import GridSpec

# Optional FFT acceleration with pyFFTW
try:
    import pyfftw

    _HAS_PYFFTW = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_PYFFTW = False
    pyfftw = None

class PFCEvolver:
    def __init__(
        self,
        grid: GridSpec,
        params: PFCParameters,
        dt: float = 1e-3,
        extra_mu: Callable[[Array], Array] | None = None,
        clip: float | None = None,
        use_pyfftw: bool = True,
        fft_threads: int | None = None,
        scheme: Literal["explicit", "semi-implicit"] = "semi-implicit",
    ) -> None:
        self.grid = grid
        self.params = params
        self.dt = dt
        self.extra_mu = extra_mu  # optional external coupling term μ_extra(ψ)
        self.clip = clip  # optional soft guard; None disables clipping
        self.fft_threads = fft_threads or max(1, (os.cpu_count() or 1) // 2)
        self.use_pyfftw = use_pyfftw and _HAS_PYFFTW
        self.scheme = scheme
        if self.use_pyfftw:
            pyfftw.interfaces.cache.enable()
            self._fftn = lambda a: pyfftw.interfaces.numpy_fft.fftn(a, threads=self.fft_threads)
            self._ifftn = lambda a: pyfftw.interfaces.numpy_fft.ifftn(a, threads=self.fft_threads)
        else:
            self._fftn = np.fft.fftn
            self._ifftn = np.fft.ifftn
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
        psi_hat = self._fftn(psi)
        # Swift-Hohenberg 算子
        operator = self.params.r + (self.params.q0**2 - self.k2) ** 2
        linear = self._ifftn(operator * psi_hat).real
        nonlinear = self.params.u * psi**3
        mu = linear + nonlinear
        if self.extra_mu is not None:
            mu = mu + self.extra_mu(psi)
        return mu

    def step(self, psi: Array) -> Array:
        psi_hat = self._fftn(psi)
        # Swift-Hohenberg 线性算子（傅里叶域）
        linear_coeff = self.params.r + (self.params.q0**2 - self.k2) ** 2
        if self.scheme == "explicit":
            mu = self.chemical_potential(psi)
            mu_hat = self._fftn(mu)
            # 守恒型动力学
            update = -self.k2 * mu_hat
            psi_new_hat = psi_hat + self.dt * update
        elif self.scheme == "semi-implicit":
            # Treat linear part implicitly, nonlinear/extra explicitly.
            nonlinear = self.params.u * psi**3
            extra = self.extra_mu(psi) if self.extra_mu is not None else 0.0
            nonlinear_hat = self._fftn(nonlinear + extra)
            denom = 1.0 + self.dt * self.k2 * linear_coeff
            update = self.dt * self.k2 * nonlinear_hat
            psi_new_hat = (psi_hat - update) / denom
        else:
            raise ValueError(f"Unknown PFC scheme: {self.scheme}")
        psi_new = self._ifftn(psi_new_hat).real
        if self.clip is not None:
            psi_new = np.clip(psi_new, -self.clip, self.clip)
            psi_new = np.nan_to_num(psi_new, nan=0.0, posinf=self.clip, neginf=-self.clip)
        else:
            psi_new = np.nan_to_num(psi_new, nan=0.0)
        return psi_new


__all__ = ["PFCEvolver"]
