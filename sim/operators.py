"""
Hybrid FFT/finite-difference operators for the ductile-fracture solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .energy import Array, CopperParameters


@dataclass(frozen=True)
class GridSpec:
    shape: Tuple[int, ...]
    spacing: Tuple[float, ...]
    periodic: Tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.spacing) or len(self.shape) != len(self.periodic):
            raise ValueError("GridSpec lengths must match.")


class FFTDifferentiator:
    """
    FFT-based derivative calculator along periodic axes.
    """

    def __init__(self, grid: GridSpec) -> None:
        self.grid = grid
        self.k_axes = self._wave_numbers()

    def _wave_numbers(self) -> Tuple[Array, ...]:
        k_axes = []
        for n, dx, periodic in zip(self.grid.shape, self.grid.spacing, self.grid.periodic):
            if not periodic:
                k_axes.append(None)
                continue
            freq = 2 * np.pi * np.fft.fftfreq(n, d=dx)
            k_axes.append(freq)
        return tuple(k_axes)

    def gradient(self, field: Array) -> Tuple[Array, ...]:
        grads = []
        f_hat = np.fft.fftn(field)
        for axis, k in enumerate(self.k_axes):
            if k is None:
                grads.append(np.gradient(field, self.grid.spacing[axis], axis=axis))
                continue
            shape = [1] * field.ndim
            shape[axis] = -1
            factor = 1j * k.reshape(shape)
            grad_hat = factor * f_hat
            grads.append(np.fft.ifftn(grad_hat).real)
        return tuple(grads)


class HybridElasticOperator:
    """
    Computes elastic strain/stress fields using FFT in periodic directions and
    central differences elsewhere.
    """

    def __init__(self, grid: GridSpec, material: CopperParameters) -> None:
        self.grid = grid
        self.material = material
        self.fft = FFTDifferentiator(grid)

    def strain_from_displacement(self, displacement: Array) -> Array:
        grads = np.stack(self.fft.gradient(displacement), axis=-1)
        strain = 0.5 * (grads + np.swapaxes(grads, -2, -1))
        return strain

    def stress_from_strain(self, strain: Array) -> Array:
        lam, mu = (
            self.material.poisson_ratio
            * self.material.youngs_modulus
            / ((1 + self.material.poisson_ratio) * (1 - 2 * self.material.poisson_ratio)),
            self.material.shear_modulus,
        )
        trace = np.trace(strain, axis1=-2, axis2=-1)[..., None, None]
        identity = np.eye(strain.shape[-1])
        return 2 * mu * (strain - identity * trace / 3) + lam * trace * identity

    def apply_uniaxial_strain(
        self,
        axis: int,
        magnitude: float,
    ) -> Tuple[Array, Array]:
        disp = np.zeros(self.grid.shape + (len(self.grid.shape),))
        coords = np.linspace(0, (self.grid.shape[axis] - 1) * self.grid.spacing[axis], self.grid.shape[axis])
        ramp = magnitude * coords / coords[-1]
        disp_component = np.zeros_like(disp[..., axis])
        disp_component = np.broadcast_to(ramp.reshape([-1 if i == axis else 1 for i in range(disp.ndim - 1)]), disp_component.shape)
        disp[..., axis] = disp_component
        strain = self.strain_from_displacement(disp)
        stress = self.stress_from_strain(strain)
        return strain, stress


__all__ = ["GridSpec", "FFTDifferentiator", "HybridElasticOperator"]
