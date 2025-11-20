"""
Mechanical equilibrium solver for anisotropic elasticity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .energy import Array, CopperParameters
from .operators import GridSpec


def periodic_gradient(field: Array, axis: int, spacing: float) -> Array:
    return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2.0 * spacing)


def sym_grad(displacement: Array, spacing: Tuple[float, ...]) -> Array:
    grad = np.zeros(displacement.shape[:-1] + (3, 3))
    for i in range(3):
        for j in range(3):
            grad[..., j, i] = periodic_gradient(displacement[..., i], j, spacing[j])
    return 0.5 * (grad + np.swapaxes(grad, -2, -1))


def divergence(stress: Array, spacing: Tuple[float, ...]) -> Array:
    div = np.zeros(stress.shape[:-1])
    for j in range(3):
        div += periodic_gradient(stress[..., :, j], j, spacing[j])
    return div


@dataclass
class MechanicalConfig:
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
        reshaped = orientation_field.reshape(-1, 3, 3)
        base = material.stiffness_tensor()
        rotated = np.einsum("npi,nqj,nrk,nsl,pqrs->nijkl", reshaped, reshaped, reshaped, reshaped, base, optimize=True)
        self.stiffness = rotated.reshape(grid.shape + (3, 3, 3, 3))
        self.num_dofs = np.prod(grid.shape) * 3
        self.grid = grid
        self.spacing = grid.spacing

    def solve(
        self,
        displacement: Array,
        crack: Array,
        macro_strain: Tuple[float, float, float],
    ) -> Tuple[Array, Array, Array]:
        mask = (1.0 - crack)[..., None, None]
        macro = np.zeros(crack.shape + (3, 3))
        for i in range(3):
            macro[..., i, i] = macro_strain[i]

        def matvec(vec: np.ndarray) -> np.ndarray:
            u = vec.reshape(crack.shape + (3,))
            strain = sym_grad(u, self.spacing)
            stress = np.einsum("...ijkl,...kl->...ij", self.stiffness, strain, optimize=True)
            stress *= mask
            divsigma = divergence(stress, self.spacing)
            return divsigma.reshape(-1)

        rhs_stress = np.einsum("...ijkl,...kl->...ij", self.stiffness, macro, optimize=True)
        rhs_stress *= mask
        rhs = -divergence(rhs_stress, self.spacing).reshape(-1)
        linop = LinearOperator((self.num_dofs, self.num_dofs), matvec)
        u0 = displacement.reshape(-1)
        solution, info = cg(linop, rhs, x0=u0, tol=self.config.tol, maxiter=self.config.max_iters)
        if info != 0:
            u = displacement
        else:
            u = solution.reshape(crack.shape + (3,))

        total_strain = sym_grad(u, self.spacing) + macro
        stress = np.einsum("...ijkl,...kl->...ij", self.stiffness, total_strain, optimize=True)
        stress *= mask
        return u, total_strain, stress


__all__ = ["MechanicalEquilibriumSolver", "MechanicalConfig"]
