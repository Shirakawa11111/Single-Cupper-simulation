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


def grid_gradient(field: Array, axis: int, spacing: float, periodic: bool) -> Array:
    if periodic:
        return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2.0 * spacing)
    if field.shape[axis] < 2:
        return np.zeros_like(field)
    grad = np.empty_like(field)
    if field.shape[axis] > 2:
        slc_mid = [slice(None)] * field.ndim
        slc_mid[axis] = slice(1, -1)
        slc_plus = slc_mid.copy()
        slc_plus[axis] = slice(2, None)
        slc_minus = slc_mid.copy()
        slc_minus[axis] = slice(None, -2)
        grad[tuple(slc_mid)] = (field[tuple(slc_plus)] - field[tuple(slc_minus)]) / (2.0 * spacing)
    slc0 = [slice(None)] * field.ndim
    slc1 = [slice(None)] * field.ndim
    slc0[axis] = 0
    slc1[axis] = 1
    grad[tuple(slc0)] = (field[tuple(slc1)] - field[tuple(slc0)]) / spacing
    slc_end = [slice(None)] * field.ndim
    slc_endm1 = [slice(None)] * field.ndim
    slc_end[axis] = -1
    slc_endm1[axis] = -2
    grad[tuple(slc_end)] = (field[tuple(slc_end)] - field[tuple(slc_endm1)]) / spacing
    return grad


def periodic_gradient(field: Array, axis: int, spacing: float) -> Array:
    return grid_gradient(field, axis, spacing, periodic=True)


def sym_grad(displacement: Array, spacing: Tuple[float, ...], periodic: Tuple[bool, ...]) -> Array:
    grad = np.zeros(displacement.shape[:-1] + (3, 3))
    for i in range(3):
        for j in range(3):
            grad[..., j, i] = grid_gradient(displacement[..., i], j, spacing[j], periodic[j])
    return 0.5 * (grad + np.swapaxes(grad, -2, -1))


def divergence(stress: Array, spacing: Tuple[float, ...], periodic: Tuple[bool, ...]) -> Array:
    div = np.zeros(stress.shape[:-1])
    for j in range(3):
        div += grid_gradient(stress[..., :, j], j, spacing[j], periodic[j])
    return div


@dataclass
class MechanicalConfig:
    max_iters: int = 200
    tol: float = 1e-5
    unilateral: bool = True
    outer_max_iters: int = 5
    outer_tol: float = 1e-6


class MechanicalEquilibriumSolver:
    def __init__(
        self,
        grid: GridSpec,
        material: CopperParameters,
        orientation_field: Array,
        fracture_k: float | None = None,
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
        self.fracture_k = fracture_k if fracture_k is not None else getattr(material, "residual_stiffness", 1e-6)

    @staticmethod
    def _split_positive(strain: Array) -> tuple[Array, Array, Array]:
        """
        Return eigenvectors (columns), positive-mask, and strain_positive for a symmetric tensor field.
        """
        vals, vecs = np.linalg.eigh(strain)
        mask = (vals > 0).astype(float)
        strain_pos = np.zeros_like(strain)
        for i in range(3):
            vi = vecs[..., i]
            outer = np.einsum("...a,...b->...ab", vi, vi)
            strain_pos += np.clip(vals[..., i], 0.0, None)[..., None, None] * outer
        return vecs, mask, strain_pos

    @staticmethod
    def _project_positive(strain: Array, vecs: Array, mask: Array) -> Array:
        """
        Linearized projector using frozen eigenvectors and sign mask.
        """
        diag = np.einsum("...ai,...ab,...bi->...i", vecs, strain, vecs, optimize=True)
        diag_pos = diag * mask
        strain_pos = np.zeros_like(strain)
        for i in range(3):
            vi = vecs[..., i]
            outer = np.einsum("...a,...b->...ab", vi, vi)
            strain_pos += diag_pos[..., i][..., None, None] * outer
        return strain_pos

    def solve(
        self,
        displacement: Array,
        crack: Array,
        macro_strain: Tuple[float, float, float],
        plastic_strain: Array | None = None,
    ) -> Tuple[Array, Array, Array]:
        # Use quadratic degradation consistent with energy; use fracture_k as residual stiffness
        k_res = self.fracture_k
        g_mask = ((1.0 - crack) ** 2 + k_res)[..., None, None]
        macro = np.zeros(crack.shape + (3, 3))
        for i in range(3):
            macro[..., i, i] = macro_strain[i]
        if plastic_strain is None:
            plastic_strain = np.zeros_like(macro)

        def _build_rhs(vecs_cached: Array, mask_cached: Array) -> Array:
            strain_eff_macro = macro - plastic_strain
            strain_macro_pos = self._project_positive(strain_eff_macro, vecs_cached, mask_cached)
            strain_macro_neg = strain_eff_macro - strain_macro_pos
            stress_rhs = g_mask * np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_macro_pos, optimize=True)
            stress_rhs += np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_macro_neg, optimize=True)
            return -divergence(stress_rhs, self.spacing, self.grid.periodic).reshape(-1)

        if not self.config.unilateral:
            def matvec(vec: np.ndarray) -> np.ndarray:
                u_loc = vec.reshape(crack.shape + (3,))
                strain = sym_grad(u_loc, self.spacing, self.grid.periodic)
                strain_eff = strain - plastic_strain
                stress = np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_eff, optimize=True)
                stress *= g_mask
                divsigma = divergence(stress, self.spacing, self.grid.periodic)
                return divsigma.reshape(-1)

            rhs_stress = np.einsum("...ijkl,...kl->...ij", self.stiffness, macro - plastic_strain, optimize=True)
            rhs_stress *= g_mask
            rhs = -divergence(rhs_stress, self.spacing, self.grid.periodic).reshape(-1)
            linop = LinearOperator((self.num_dofs, self.num_dofs), matvec)
            u0 = displacement.reshape(-1)
            solution, info = cg(linop, rhs, x0=u0, rtol=self.config.tol, atol=0.0, maxiter=self.config.max_iters)
            u = displacement if info != 0 else solution.reshape(crack.shape + (3,))
        else:
            u = displacement.copy()
            for _ in range(self.config.outer_max_iters):
                strain_total = sym_grad(u, self.spacing, self.grid.periodic) + macro
                strain_eff = strain_total - plastic_strain
                vecs_cached, mask_cached, _ = self._split_positive(strain_eff)

                def matvec(vec: np.ndarray) -> np.ndarray:
                    u_loc = vec.reshape(crack.shape + (3,))
                    strain = sym_grad(u_loc, self.spacing, self.grid.periodic)
                    strain_eff_loc = strain - plastic_strain
                    strain_pos = self._project_positive(strain_eff_loc, vecs_cached, mask_cached)
                    strain_neg = strain_eff_loc - strain_pos
                    stress = g_mask * np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_pos, optimize=True)
                    stress += np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_neg, optimize=True)
                    divsigma = divergence(stress, self.spacing, self.grid.periodic)
                    return divsigma.reshape(-1)

                rhs = _build_rhs(vecs_cached, mask_cached)
                linop = LinearOperator((self.num_dofs, self.num_dofs), matvec)
                u0 = u.reshape(-1)
                solution, info = cg(linop, rhs, x0=u0, rtol=self.config.tol, atol=0.0, maxiter=self.config.max_iters)
                u_new = u if info != 0 else solution.reshape(crack.shape + (3,))
                norm_prev = np.linalg.norm(u.reshape(-1)) + 1e-16
                if np.linalg.norm(u_new.reshape(-1) - u.reshape(-1)) / norm_prev < self.config.outer_tol:
                    u = u_new
                    break
                u = u_new

        total_strain = sym_grad(u, self.spacing, self.grid.periodic) + macro
        strain_eff = total_strain - plastic_strain
        if self.config.unilateral:
            vecs_final, mask_final, strain_pos = self._split_positive(strain_eff)
            strain_neg = strain_eff - strain_pos
            stress = g_mask * np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_pos, optimize=True)
            stress += np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_neg, optimize=True)
        else:
            stress = np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_eff, optimize=True)
            stress *= g_mask
        return u, total_strain, stress


__all__ = ["MechanicalEquilibriumSolver", "MechanicalConfig"]
