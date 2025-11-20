"""
Spectral Phase-Field Crystal evolver.
"""

from __future__ import annotations

import numpy as np

from .energy import Array, PFCParameters
from .operators import GridSpec


class PFCEvolver:
    def __init__(self, grid: GridSpec, params: PFCParameters, dt: float = 1e-3) -> None:
        self.grid = grid
        self.params = params
        self.dt = dt
        self.k2 = self._wave_numbers_squared()

    def _wave_numbers_squared(self) -> Array:
        axes = []
        for n, d in zip(self.grid.shape, self.grid.spacing):
            k = 2 * np.pi * np.fft.fftfreq(n, d=d)
            axes.append(k)
        k2 = np.zeros(self.grid.shape)
        for axis, k in enumerate(axes):
            shape = [1] * len(self.grid.shape)
            shape[axis] = -1
            k2 += k.reshape(shape) ** 2
        return k2

    def chemical_potential(self, psi: Array) -> Array:
        psi_hat = np.fft.fftn(psi)
        operator = self.params.r + (self.params.q0**2 - self.k2) ** 2
        linear = np.fft.ifftn(operator * psi_hat).real
        nonlinear = self.params.u * psi**3
        return linear + nonlinear

    def step(self, psi: Array) -> Array:
        mu = self.chemical_potential(psi)
        mu_hat = np.fft.fftn(mu)
        psi_hat = np.fft.fftn(psi)
        update = -self.k2 * mu_hat
        psi_new_hat = psi_hat + self.dt * update
        psi_new = np.fft.ifftn(psi_new_hat).real
        psi_new = np.clip(psi_new, -1.0, 1.0)
        return psi_new


__all__ = ["PFCEvolver"]
