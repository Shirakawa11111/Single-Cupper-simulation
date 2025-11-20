"""
Mechanical equilibrium solver for anisotropic elasticity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .energy import Array, CopperParameters
from .operators import GridSpec


def sym_grad(displacement: Array, spacing: Tuple[float, ...]) -> Array:
    grad = np.zeros(displacement.shape + (len(spacing),))
    for axis, h in enumerate(spacing):
        grad[..., axis] = np.gradient(displacement, h, axis=axis)
    return 0.5 * (grad + np.swapaxes(grad, -2, -1))


def divergence(stress: Array, spacing: Tuple[float, ...]) -> Array:
    components = []
    for i in range(stress.shape[-2]):
        div_i = 0.0
        for j in range(stress.shape[-1]):
            div_i += np.gradient(stress[..., i, j], spacing[j], axis=j)
        components.append(div_i)
    return np.stack(components, axis=-1)


@dataclass
class MechanicalConfig:
    step_size: float = 1e-3
    max_iters: int = 200
    tol: float = 1e-5


class MechanicalEquilibriumSolver:
    def __init__(
        self,
        grid: GridSpec,
        material: CopperParameters,
        orientation_field: Array,
        config: MechanicalConfig | None = None,
    ) -> None:
        self.grid = grid
        self.material = material
        self.config = config or MechanicalConfig()
        self.stiffness = np.zeros(grid.shape + (3, 3, 3, 3))
        for idx in np.ndindex(grid.shape):
            rot = orientation_field[idx]
            self.stiffness[idx] = material.stiffness_tensor(rot)

    def apply_boundary(self, displacement: Array, macro_strain: Tuple[float, float, float]) -> None:
        L = [n * d for n, d in zip(self.grid.shape, self.grid.spacing)]
        for axis, eps in enumerate(macro_strain):
            delta = eps * L[axis]
            slicer_top = [slice(None)] * displacement.ndim
            slicer_bottom = [slice(None)] * displacement.ndim
            slicer_top[axis] = -1
            slicer_bottom[axis] = 0
            displacement[tuple(slicer_bottom)][..., axis] = 0.0
            displacement[tuple(slicer_top)][..., axis] = delta

    def solve(
        self,
        displacement: Array,
        crack: Array,
        macro_strain: Tuple[float, float, float],
    ) -> Tuple[Array, Array, Array]:
        u = displacement.copy()
        self.apply_boundary(u, macro_strain)
        for _ in range(self.config.max_iters):
            strain = sym_grad(u, self.grid.spacing)
            stress = np.einsum("...ijkl,...kl->...ij", self.stiffness, strain, optimize=True)
            stress *= (1.0 - crack)[..., None, None]
            res = divergence(stress, self.grid.spacing)
            max_res = np.max(np.abs(res))
            if max_res < self.config.tol:
                break
            u += self.config.step_size * res
            self.apply_boundary(u, macro_strain)
        strain = sym_grad(u, self.grid.spacing)
        stress = np.einsum("...ijkl,...kl->...ij", self.stiffness, strain, optimize=True)
        stress *= (1.0 - crack)[..., None, None]
        return u, strain, stress


__all__ = ["MechanicalEquilibriumSolver", "MechanicalConfig"]
