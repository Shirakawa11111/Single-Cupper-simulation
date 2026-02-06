"""
Mechanical equilibrium solver for anisotropic elasticity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal

import numpy as np
import warnings
from scipy.sparse.linalg import LinearOperator, cg, gmres

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
    unilateral_mode: Literal["spectral", "volumetric"] = "spectral"
    outer_max_iters: int = 5
    outer_tol: float = 1e-6
    # Small linear regularization added to operator to reduce singularity issues.
    regularization: float = 1e-10
    # Accept iterative result when relative residual is below this threshold.
    accept_rel_residual: float = 5e-4
    # Accept finite CG result even when info>0 (incomplete convergence),
    # to avoid hard stalls in singular/near-singular regimes.
    accept_incomplete_cg: bool = False
    # Reject iterative solutions with unrealistic absolute magnitude.
    solution_abs_limit: float = 10.0
    # Fallback to GMRES when CG fails residual acceptance.
    enable_gmres_fallback: bool = False
    gmres_restart: int = 40
    gmres_maxiter: int = 60


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
        self.last_solve_info: dict[str, int | float | bool | str] = {
            "unilateral": bool(self.config.unilateral),
            "mode": self.config.unilateral_mode if self.config.unilateral else "linear",
            "cg_failures": 0,
            "last_cg_info": 0,
            "outer_iterations": 0,
            "outer_converged": True,
            "rel_change": 0.0,
            "solver_used": "cg",
            "accepted": True,
            "rel_residual": 0.0,
            "runtime_warning_count": 0,
            "gmres_fallback_used": False,
        }

    @staticmethod
    def _relative_residual(linop: LinearOperator, x: np.ndarray, b: np.ndarray) -> float:
        ax = linop.matvec(x)
        if not np.all(np.isfinite(ax)):
            return float("inf")
        res = ax - b
        num = float(np.linalg.norm(res))
        den = float(np.linalg.norm(b)) + 1e-16
        if not np.isfinite(num):
            return float("inf")
        return num / den

    @staticmethod
    def _remove_rigid_translation(u: Array) -> Array:
        mean_vec = np.mean(u.reshape(-1, 3), axis=0, keepdims=True)
        return u - mean_vec.reshape((1, 1, 1, 3))

    def _iterative_solve(
        self,
        linop: LinearOperator,
        rhs: np.ndarray,
        x0: np.ndarray,
    ) -> tuple[np.ndarray, int, float, int, str, bool]:
        rhs_norm = float(np.linalg.norm(rhs))
        if (not np.isfinite(rhs_norm)) or rhs_norm < 1e-14:
            return x0.copy(), 0, 0.0, 0, "rhs_skip", True

        warn_count = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            x_cg, info_cg = cg(
                linop,
                rhs,
                x0=x0,
                rtol=self.config.tol,
                atol=0.0,
                maxiter=self.config.max_iters,
            )
        warn_count += int(sum(1 for w in caught if issubclass(w.category, RuntimeWarning)))

        cg_finite = bool(np.all(np.isfinite(x_cg)))
        if cg_finite:
            cg_finite = float(np.max(np.abs(x_cg))) <= float(self.config.solution_abs_limit)
        rel_cg = self._relative_residual(linop, x_cg, rhs) if cg_finite else float("inf")
        cg_incomplete_ok = (
            self.config.accept_incomplete_cg
            and info_cg > 0
            and cg_finite
            and np.isfinite(rel_cg)
        )
        cg_ok = (info_cg == 0 or rel_cg <= self.config.accept_rel_residual or cg_incomplete_ok) and cg_finite
        if cg_ok:
            method = "cg_incomplete" if cg_incomplete_ok and info_cg != 0 else "cg"
            return x_cg, int(info_cg), float(rel_cg), warn_count, method, True

        if self.config.enable_gmres_fallback:
            restart = max(5, int(self.config.gmres_restart))
            maxiter = max(1, int(self.config.gmres_maxiter))
            x_seed = x_cg if cg_finite else x0
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                x_gmres, info_gmres = gmres(
                    linop,
                    rhs,
                    x0=x_seed,
                    rtol=self.config.tol,
                    atol=0.0,
                    restart=restart,
                    maxiter=maxiter,
                )
            warn_count += int(sum(1 for w in caught if issubclass(w.category, RuntimeWarning)))
            gmres_finite = bool(np.all(np.isfinite(x_gmres)))
            if gmres_finite:
                gmres_finite = float(np.max(np.abs(x_gmres))) <= float(self.config.solution_abs_limit)
            rel_gmres = self._relative_residual(linop, x_gmres, rhs) if gmres_finite else float("inf")
            gmres_ok = (info_gmres == 0 or rel_gmres <= self.config.accept_rel_residual) and gmres_finite
            if gmres_ok:
                return x_gmres, int(info_gmres), float(rel_gmres), warn_count, "gmres", True
            return x0.copy(), int(info_gmres), float(rel_gmres), warn_count, "hold", False

        return x0.copy(), int(info_cg), float(rel_cg), warn_count, "hold", False

    @staticmethod
    def _split_positive_spectral(strain: Array) -> tuple[Array, Array, Array]:
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
    def _project_positive_spectral(strain: Array, vecs: Array, mask: Array) -> Array:
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

    @staticmethod
    def _split_positive_volumetric(strain: Array) -> tuple[Array, Array, Array]:
        """
        Volumetric/deviatoric split: tensile volumetric part + full deviatoric.
        Returns mask (tr>0), strain_pos, strain_neg.
        """
        tr = np.trace(strain, axis1=-2, axis2=-1)
        mask = (tr > 0).astype(float)
        tr_pos = np.clip(tr, 0.0, None)
        tr_neg = tr - tr_pos
        eye = np.eye(3)
        dev = strain - (tr[..., None, None] / 3.0) * eye
        strain_pos = dev + (tr_pos[..., None, None] / 3.0) * eye
        strain_neg = (tr_neg[..., None, None] / 3.0) * eye
        return mask, strain_pos, strain_neg

    @staticmethod
    def _project_positive_volumetric(strain: Array, mask: Array) -> Array:
        """
        Linearized volumetric projector using frozen sign of tr(strain).
        """
        tr = np.trace(strain, axis1=-2, axis2=-1)
        tr_pos = tr * mask
        eye = np.eye(3)
        dev = strain - (tr[..., None, None] / 3.0) * eye
        return dev + (tr_pos[..., None, None] / 3.0) * eye

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
        displacement = np.nan_to_num(displacement, nan=0.0, posinf=0.0, neginf=0.0)
        plastic_strain = np.nan_to_num(plastic_strain, nan=0.0, posinf=0.0, neginf=0.0)
        cg_failures = 0
        last_cg_info = 0
        outer_iterations = 0
        outer_converged = True
        rel_change = 0.0
        max_rhs_norm = 0.0
        runtime_warning_count = 0
        solver_used = "cg"
        accepted = True
        rel_residual = 0.0
        gmres_fallback_used = False

        if not self.config.unilateral:
            def matvec(vec: np.ndarray) -> np.ndarray:
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                u_loc = vec.reshape(crack.shape + (3,))
                strain = sym_grad(u_loc, self.spacing, self.grid.periodic)
                strain_eff = np.nan_to_num(strain - plastic_strain, nan=0.0, posinf=0.0, neginf=0.0)
                stress = np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_eff, optimize=True)
                stress *= g_mask
                divsigma = divergence(stress, self.spacing, self.grid.periodic)
                if self.config.regularization > 0.0:
                    divsigma += self.config.regularization * u_loc
                return divsigma.reshape(-1)

            rhs_stress = np.einsum("...ijkl,...kl->...ij", self.stiffness, macro - plastic_strain, optimize=True)
            rhs_stress *= g_mask
            rhs = -divergence(rhs_stress, self.spacing, self.grid.periodic).reshape(-1)
            linop = LinearOperator((self.num_dofs, self.num_dofs), matvec)
            u0 = displacement.reshape(-1)
            rhs_norm = float(np.linalg.norm(rhs))
            max_rhs_norm = max(max_rhs_norm, rhs_norm)
            solution, info, rel_residual, warn_n, solver_used, accepted = self._iterative_solve(linop, rhs, u0)
            runtime_warning_count += int(warn_n)
            gmres_fallback_used = solver_used == "gmres"
            last_cg_info = int(info)
            if not accepted:
                cg_failures += 1
            outer_iterations = 1
            outer_converged = bool(accepted)
            u = self._remove_rigid_translation(solution.reshape(crack.shape + (3,)))
        else:
            u = displacement.copy()
            outer_converged = False
            for outer_idx in range(self.config.outer_max_iters):
                outer_iterations = outer_idx + 1
                strain_total = sym_grad(u, self.spacing, self.grid.periodic) + macro
                strain_eff = strain_total - plastic_strain
                if self.config.unilateral_mode == "spectral":
                    vecs_cached, mask_cached, _ = self._split_positive_spectral(strain_eff)

                    def project_positive(strain_loc: Array) -> Array:
                        return self._project_positive_spectral(strain_loc, vecs_cached, mask_cached)

                elif self.config.unilateral_mode == "volumetric":
                    mask_cached, _, _ = self._split_positive_volumetric(strain_eff)

                    def project_positive(strain_loc: Array) -> Array:
                        return self._project_positive_volumetric(strain_loc, mask_cached)

                else:
                    raise ValueError(f"Unknown unilateral_mode: {self.config.unilateral_mode}")

                def _build_rhs() -> Array:
                    strain_eff_macro = macro - plastic_strain
                    strain_macro_pos = project_positive(strain_eff_macro)
                    strain_macro_neg = strain_eff_macro - strain_macro_pos
                    stress_rhs = g_mask * np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_macro_pos, optimize=True)
                    stress_rhs += np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_macro_neg, optimize=True)
                    return -divergence(stress_rhs, self.spacing, self.grid.periodic).reshape(-1)

                def matvec(vec: np.ndarray) -> np.ndarray:
                    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                    u_loc = vec.reshape(crack.shape + (3,))
                    strain = sym_grad(u_loc, self.spacing, self.grid.periodic)
                    strain_eff_loc = np.nan_to_num(strain - plastic_strain, nan=0.0, posinf=0.0, neginf=0.0)
                    strain_pos = project_positive(strain_eff_loc)
                    strain_neg = strain_eff_loc - strain_pos
                    stress = g_mask * np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_pos, optimize=True)
                    stress += np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_neg, optimize=True)
                    divsigma = divergence(stress, self.spacing, self.grid.periodic)
                    if self.config.regularization > 0.0:
                        divsigma += self.config.regularization * u_loc
                    return divsigma.reshape(-1)

                rhs = _build_rhs()
                linop = LinearOperator((self.num_dofs, self.num_dofs), matvec)
                u0 = u.reshape(-1)
                rhs_norm = float(np.linalg.norm(rhs))
                max_rhs_norm = max(max_rhs_norm, rhs_norm)
                solution, info, rel_residual, warn_n, solver_used, accepted = self._iterative_solve(linop, rhs, u0)
                runtime_warning_count += int(warn_n)
                gmres_fallback_used = gmres_fallback_used or solver_used == "gmres"
                last_cg_info = int(info)
                if not accepted:
                    cg_failures += 1
                u_new = self._remove_rigid_translation(solution.reshape(crack.shape + (3,)))
                if not accepted:
                    rel_change = float("inf")
                    u = u_new
                    continue
                norm_prev = np.linalg.norm(u.reshape(-1)) + 1e-16
                rel_change = np.linalg.norm(u_new.reshape(-1) - u.reshape(-1)) / norm_prev
                if rel_change < self.config.outer_tol:
                    u = u_new
                    outer_converged = True
                    break
                u = u_new

        self.last_solve_info = {
            "unilateral": bool(self.config.unilateral),
            "mode": self.config.unilateral_mode if self.config.unilateral else "linear",
            "cg_failures": int(cg_failures),
            "last_cg_info": int(last_cg_info),
            "outer_iterations": int(outer_iterations),
            "outer_converged": bool(outer_converged),
            "rel_change": float(rel_change),
            "rhs_norm_max": float(max_rhs_norm),
            "solver_used": solver_used,
            "accepted": bool(accepted),
            "rel_residual": float(rel_residual),
            "runtime_warning_count": int(runtime_warning_count),
            "gmres_fallback_used": bool(gmres_fallback_used),
        }

        total_strain = sym_grad(u, self.spacing, self.grid.periodic) + macro
        strain_eff = total_strain - plastic_strain
        if self.config.unilateral:
            if self.config.unilateral_mode == "spectral":
                _, _, strain_pos = self._split_positive_spectral(strain_eff)
                strain_neg = strain_eff - strain_pos
            elif self.config.unilateral_mode == "volumetric":
                _, strain_pos, strain_neg = self._split_positive_volumetric(strain_eff)
            else:
                raise ValueError(f"Unknown unilateral_mode: {self.config.unilateral_mode}")
            stress = g_mask * np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_pos, optimize=True)
            stress += np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_neg, optimize=True)
        else:
            stress = np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_eff, optimize=True)
            stress *= g_mask
        return u, total_strain, stress


__all__ = ["MechanicalEquilibriumSolver", "MechanicalConfig"]
