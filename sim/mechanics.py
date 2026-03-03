"""
Mechanical equilibrium solver for anisotropic elasticity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal, Callable

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
    # If True, allow accepting finite incomplete CG solution even when residual is non-finite.
    accept_incomplete_without_residual: bool = False
    # Reject iterative solutions with unrealistic absolute magnitude.
    solution_abs_limit: float = 10.0
    # When True, clip solution to `solution_abs_limit` instead of rejecting it.
    clip_solution_on_limit: bool = False
    # Fallback to GMRES when CG fails residual acceptance.
    enable_gmres_fallback: bool = False
    gmres_restart: int = 40
    gmres_maxiter: int = 60
    # Optional linear preconditioner for iterative solves.
    preconditioner: Literal["none", "jacobi"] = "none"
    preconditioner_floor: float = 1e-5
    preconditioner_g_min: float = 5e-2
    # Displacement-driven loading (Dirichlet penalty) on x-min/x-max boundaries.
    displacement_bc_x: bool = False
    displacement_bc_penalty: float = 1e6
    displacement_bc_anchor_lateral: bool = True
    displacement_bc_hard: bool = False
    # Solve mechanics in normalized coordinates (x*=x/Lref, u*=u/Lref) for better conditioning.
    nondim_kinematics: bool = False


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
            "displacement_bc_x": bool(self.config.displacement_bc_x),
            "displacement_bc_hard": bool(self.config.displacement_bc_hard),
            "nondim_kinematics": bool(self.config.nondim_kinematics),
            "length_ref": 1.0,
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
    def _safe_l2_norm(vec: np.ndarray) -> float:
        arr = np.asarray(vec, dtype=float)
        if arr.size == 0:
            return 0.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0e200, neginf=-1.0e200)
        max_abs = float(np.max(np.abs(arr)))
        if (not np.isfinite(max_abs)) or max_abs == 0.0:
            return 0.0 if max_abs == 0.0 else float("inf")
        scaled = arr / max_abs
        nrm = max_abs * float(np.linalg.norm(scaled))
        return nrm if np.isfinite(nrm) else float("inf")

    @staticmethod
    def _relative_residual(linop: LinearOperator, x: np.ndarray, b: np.ndarray) -> float:
        ax = np.nan_to_num(
            linop.matvec(x),
            nan=0.0,
            posinf=1.0e200,
            neginf=-1.0e200,
        )
        b_safe = np.nan_to_num(b, nan=0.0, posinf=1.0e200, neginf=-1.0e200)
        res = ax - b_safe
        num = MechanicalEquilibriumSolver._safe_l2_norm(res)
        den = MechanicalEquilibriumSolver._safe_l2_norm(b_safe) + 1e-16
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
        sanitize: Callable[[np.ndarray], np.ndarray] | None = None,
        preconditioner: LinearOperator | None = None,
        solution_abs_limit: float | None = None,
    ) -> tuple[np.ndarray, int, float, int, str, bool, bool]:
        sanitize_fn = sanitize or (lambda v: v)
        x0_eval = sanitize_fn(np.nan_to_num(x0.copy(), nan=0.0, posinf=0.0, neginf=0.0))
        rhs_norm = self._safe_l2_norm(rhs)
        if (not np.isfinite(rhs_norm)) or rhs_norm < 1e-14:
            return x0_eval, 0, 0.0, 0, "rhs_skip", True, False

        def enforce_solution_limit(vec: np.ndarray) -> tuple[np.ndarray, bool, bool]:
            arr = sanitize_fn(np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0))
            is_finite = bool(np.all(np.isfinite(arr)))
            clipped = False
            if is_finite:
                limit = float(self.config.solution_abs_limit if solution_abs_limit is None else solution_abs_limit)
                if np.isfinite(limit) and limit > 0.0:
                    max_abs = float(np.max(np.abs(arr)))
                    if max_abs > limit:
                        if self.config.clip_solution_on_limit:
                            arr = arr * (limit / (max_abs + 1e-30))
                            clipped = True
                        else:
                            is_finite = False
            return arr, is_finite, clipped

        warn_count = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            x_cg, info_cg = cg(
                linop,
                rhs,
                x0=x0_eval,
                rtol=self.config.tol,
                atol=0.0,
                maxiter=self.config.max_iters,
                M=preconditioner,
            )
        warn_count += int(sum(1 for w in caught if issubclass(w.category, RuntimeWarning)))

        x_cg_eval, cg_finite, cg_clipped = enforce_solution_limit(x_cg)
        rel_cg = self._relative_residual(linop, x_cg_eval, rhs) if cg_finite else float("inf")
        cg_incomplete_ok = (
            self.config.accept_incomplete_cg
            and info_cg > 0
            and cg_finite
            and np.isfinite(rel_cg)
        )
        cg_incomplete_relaxed = (
            self.config.accept_incomplete_cg
            and self.config.accept_incomplete_without_residual
            and info_cg > 0
            and cg_finite
        )
        cg_ok = (
            info_cg == 0
            or rel_cg <= self.config.accept_rel_residual
            or cg_incomplete_ok
            or cg_incomplete_relaxed
        ) and cg_finite
        if cg_ok:
            if cg_incomplete_ok and info_cg != 0:
                method = "cg_incomplete"
            elif cg_incomplete_relaxed and info_cg != 0:
                method = "cg_incomplete_relaxed"
            else:
                method = "cg"
            if cg_clipped:
                method = f"{method}_clip"
            return x_cg_eval, int(info_cg), float(rel_cg), warn_count, method, True, cg_clipped

        if self.config.enable_gmres_fallback:
            restart = max(5, int(self.config.gmres_restart))
            maxiter = max(1, int(self.config.gmres_maxiter))
            x_seed = x_cg_eval if cg_finite else x0_eval
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
                    M=preconditioner,
                )
            warn_count += int(sum(1 for w in caught if issubclass(w.category, RuntimeWarning)))
            x_gmres_eval, gmres_finite, gmres_clipped = enforce_solution_limit(x_gmres)
            rel_gmres = self._relative_residual(linop, x_gmres_eval, rhs) if gmres_finite else float("inf")
            gmres_ok = (info_gmres == 0 or rel_gmres <= self.config.accept_rel_residual) and gmres_finite
            if gmres_ok:
                method = "gmres_clip" if gmres_clipped else "gmres"
                return x_gmres_eval, int(info_gmres), float(rel_gmres), warn_count, method, True, gmres_clipped
            return x0_eval.copy(), int(info_gmres), float(rel_gmres), warn_count, "hold", False, False

        return x0_eval.copy(), int(info_cg), float(rel_cg), warn_count, "hold", False, False

    def _build_preconditioner(self, g_mask: Array, spacing: Tuple[float, ...] | None = None) -> LinearOperator | None:
        if self.config.preconditioner != "jacobi":
            return None
        spacing_use = spacing if spacing is not None else self.spacing
        g = np.asarray(g_mask[..., 0, 0], dtype=float)
        g = np.clip(g, self.config.preconditioner_g_min, 1.0)
        cdiag = np.stack(
            [np.abs(self.stiffness[..., i, i, i, i]) for i in range(3)],
            axis=-1,
        )
        cdiag = np.nan_to_num(cdiag, nan=1.0, posinf=1.0, neginf=1.0)
        floor = max(float(self.config.preconditioner_floor), 1e-12)
        lap_coeff = 0.0
        for dx in spacing_use:
            lap_coeff += 2.0 / max(float(dx) * float(dx), 1e-12)
        diag = float(self.config.regularization) + lap_coeff * g[..., None] * np.maximum(cdiag, floor)
        diag = np.maximum(np.nan_to_num(diag, nan=floor, posinf=floor, neginf=floor), floor)
        inv_diag = (1.0 / diag).reshape(-1)

        def apply(vec: np.ndarray) -> np.ndarray:
            v = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            return inv_diag * v

        return LinearOperator((self.num_dofs, self.num_dofs), matvec=apply)

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
        use_nondim_kin = bool(self.config.nondim_kinematics)
        length_ref = 1.0
        if use_nondim_kin:
            positive_spacings = [float(s) for s in self.spacing if np.isfinite(float(s)) and float(s) > 0.0]
            if positive_spacings:
                length_ref = min(positive_spacings)
                if (not np.isfinite(length_ref)) or length_ref <= 0.0:
                    length_ref = 1.0
                    use_nondim_kin = False
            else:
                use_nondim_kin = False
        spacing_solve = tuple(float(s) / length_ref for s in self.spacing) if use_nondim_kin else self.spacing
        # Keep numerical regularization/penalty consistent with scaled divergence operator.
        equil_scale = float(length_ref) if use_nondim_kin else 1.0
        displacement_scale = float(length_ref) if use_nondim_kin else 1.0
        # Use quadratic degradation consistent with energy; use fracture_k as residual stiffness
        k_res = self.fracture_k
        g_mask = ((1.0 - crack) ** 2 + k_res)[..., None, None]
        macro = np.zeros(crack.shape + (3, 3))
        for i in range(3):
            macro[..., i, i] = macro_strain[i]
        if plastic_strain is None:
            plastic_strain = np.zeros_like(macro)
        displacement = np.nan_to_num(displacement, nan=0.0, posinf=0.0, neginf=0.0)
        if use_nondim_kin:
            displacement = displacement / displacement_scale
        plastic_strain = np.nan_to_num(plastic_strain, nan=0.0, posinf=0.0, neginf=0.0)
        use_disp_bc = bool(self.config.displacement_bc_x) and (not self.grid.periodic[0])
        use_hard_bc = use_disp_bc and bool(self.config.displacement_bc_hard)
        disp_bc_penalty = max(float(self.config.displacement_bc_penalty), 0.0)
        bc_mask_u = np.zeros(crack.shape + (3,), dtype=float)
        bc_target = np.zeros(crack.shape + (3,), dtype=float)
        if use_disp_bc:
            lx = spacing_solve[0] * max(int(self.grid.shape[0]) - 1, 1)
            ux_right = float(macro_strain[0]) * lx
            bc_mask_u[0, :, :, 0] = 1.0
            bc_mask_u[-1, :, :, 0] = 1.0
            bc_target[-1, :, :, 0] = ux_right
            if self.config.displacement_bc_anchor_lateral:
                # Anchor lateral rigid modes on the whole x=0 face for better stability.
                bc_mask_u[0, :, :, 1] = 1.0
                bc_mask_u[0, :, :, 2] = 1.0
                bc_target[0, :, :, 1] = 0.0
                bc_target[0, :, :, 2] = 0.0
        macro_eff = np.zeros_like(macro) if use_disp_bc else macro
        bc_mask_flat = (bc_mask_u.reshape(-1) > 0.5) if use_disp_bc else np.zeros(self.num_dofs, dtype=bool)
        bc_target_flat = bc_target.reshape(-1) if use_disp_bc else np.zeros(self.num_dofs, dtype=float)
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
        preconditioner = self._build_preconditioner(g_mask, spacing=spacing_solve)
        solution_clipped = False
        solution_abs_limit_solve = (
            float(self.config.solution_abs_limit) / displacement_scale
            if use_nondim_kin and np.isfinite(float(self.config.solution_abs_limit))
            else float(self.config.solution_abs_limit)
        )

        if not self.config.unilateral:
            def sanitize_solution(vec: np.ndarray) -> np.ndarray:
                arr = vec.reshape(crack.shape + (3,))
                if not use_disp_bc:
                    arr = self._remove_rigid_translation(arr)
                return arr.reshape(-1)

            def matvec(vec: np.ndarray) -> np.ndarray:
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                u_loc = vec.reshape(crack.shape + (3,))
                strain = sym_grad(u_loc, spacing_solve, self.grid.periodic)
                strain_eff = np.nan_to_num(strain - plastic_strain, nan=0.0, posinf=0.0, neginf=0.0)
                stress = np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_eff, optimize=True)
                stress *= g_mask
                divsigma = divergence(stress, spacing_solve, self.grid.periodic)
                if self.config.regularization > 0.0:
                    divsigma += (self.config.regularization * equil_scale) * u_loc
                if use_disp_bc and (not use_hard_bc) and disp_bc_penalty > 0.0:
                    divsigma += (disp_bc_penalty * equil_scale) * bc_mask_u * u_loc
                divsigma = np.nan_to_num(divsigma, nan=0.0, posinf=0.0, neginf=0.0)
                out = divsigma.reshape(-1)
                if use_hard_bc:
                    out[bc_mask_flat] = vec[bc_mask_flat]
                return out

            macro_rhs = macro_eff
            rhs_stress = np.einsum(
                "...ijkl,...kl->...ij",
                self.stiffness,
                macro_rhs - plastic_strain,
                optimize=True,
            )
            rhs_stress *= g_mask
            rhs_arr = -divergence(rhs_stress, spacing_solve, self.grid.periodic)
            if use_disp_bc and (not use_hard_bc) and disp_bc_penalty > 0.0:
                rhs_arr += (disp_bc_penalty * equil_scale) * bc_mask_u * bc_target
            rhs = np.nan_to_num(rhs_arr, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
            if use_hard_bc:
                rhs[bc_mask_flat] = bc_target_flat[bc_mask_flat]
            linop = LinearOperator((self.num_dofs, self.num_dofs), matvec)
            u0 = displacement.reshape(-1)
            if use_hard_bc:
                u0[bc_mask_flat] = bc_target_flat[bc_mask_flat]
            rhs_norm = float(np.linalg.norm(rhs))
            max_rhs_norm = max(max_rhs_norm, rhs_norm)
            solution, info, rel_residual, warn_n, solver_used, accepted, clipped = self._iterative_solve(
                linop,
                rhs,
                u0,
                sanitize=sanitize_solution,
                preconditioner=preconditioner,
                solution_abs_limit=solution_abs_limit_solve,
            )
            solution_clipped = bool(clipped)
            runtime_warning_count += int(warn_n)
            gmres_fallback_used = str(solver_used).startswith("gmres")
            last_cg_info = int(info)
            if not accepted:
                cg_failures += 1
            outer_iterations = 1
            outer_converged = bool(accepted)
            u = solution.reshape(crack.shape + (3,))
            if not use_disp_bc:
                u = self._remove_rigid_translation(u)
        else:
            u = displacement.copy()
            if use_disp_bc:
                u[..., 0] = 0.0
                u[-1, :, :, 0] = bc_target[-1, :, :, 0]
                if self.config.displacement_bc_anchor_lateral:
                    u[0, :, :, 1] = 0.0
                    u[0, :, :, 2] = 0.0
            outer_converged = False
            def sanitize_solution(vec: np.ndarray) -> np.ndarray:
                arr = vec.reshape(crack.shape + (3,))
                if not use_disp_bc:
                    arr = self._remove_rigid_translation(arr)
                return arr.reshape(-1)
            for outer_idx in range(self.config.outer_max_iters):
                outer_iterations = outer_idx + 1
                strain_total = sym_grad(u, spacing_solve, self.grid.periodic) + macro_eff
                strain_eff = np.nan_to_num(
                    strain_total - plastic_strain,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
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
                    strain_eff_macro = macro_eff - plastic_strain
                    strain_macro_pos = project_positive(strain_eff_macro)
                    strain_macro_neg = strain_eff_macro - strain_macro_pos
                    stress_rhs = g_mask * np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_macro_pos, optimize=True)
                    stress_rhs += np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_macro_neg, optimize=True)
                    rhs_arr = -divergence(stress_rhs, spacing_solve, self.grid.periodic)
                    if use_disp_bc and (not use_hard_bc) and disp_bc_penalty > 0.0:
                        rhs_arr += (disp_bc_penalty * equil_scale) * bc_mask_u * bc_target
                    rhs_flat = np.nan_to_num(rhs_arr, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
                    if use_hard_bc:
                        rhs_flat[bc_mask_flat] = bc_target_flat[bc_mask_flat]
                    return rhs_flat

                def matvec(vec: np.ndarray) -> np.ndarray:
                    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                    u_loc = vec.reshape(crack.shape + (3,))
                    strain = sym_grad(u_loc, spacing_solve, self.grid.periodic)
                    strain_eff_loc = np.nan_to_num(strain - plastic_strain, nan=0.0, posinf=0.0, neginf=0.0)
                    strain_pos = project_positive(strain_eff_loc)
                    strain_neg = strain_eff_loc - strain_pos
                    stress = g_mask * np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_pos, optimize=True)
                    stress += np.einsum("...ijkl,...kl->...ij", self.stiffness, strain_neg, optimize=True)
                    divsigma = divergence(stress, spacing_solve, self.grid.periodic)
                    if self.config.regularization > 0.0:
                        divsigma += (self.config.regularization * equil_scale) * u_loc
                    if use_disp_bc and (not use_hard_bc) and disp_bc_penalty > 0.0:
                        divsigma += (disp_bc_penalty * equil_scale) * bc_mask_u * u_loc
                    divsigma = np.nan_to_num(divsigma, nan=0.0, posinf=0.0, neginf=0.0)
                    out = divsigma.reshape(-1)
                    if use_hard_bc:
                        out[bc_mask_flat] = vec[bc_mask_flat]
                    return out

                rhs = _build_rhs()
                linop = LinearOperator((self.num_dofs, self.num_dofs), matvec)
                u0 = u.reshape(-1)
                if use_hard_bc:
                    u0[bc_mask_flat] = bc_target_flat[bc_mask_flat]
                rhs_norm = float(np.linalg.norm(rhs))
                max_rhs_norm = max(max_rhs_norm, rhs_norm)
                solution, info, rel_residual, warn_n, solver_used, accepted, clipped = self._iterative_solve(
                    linop,
                    rhs,
                    u0,
                    sanitize=sanitize_solution,
                    preconditioner=preconditioner,
                    solution_abs_limit=solution_abs_limit_solve,
                )
                solution_clipped = solution_clipped or bool(clipped)
                runtime_warning_count += int(warn_n)
                gmres_fallback_used = gmres_fallback_used or str(solver_used).startswith("gmres")
                last_cg_info = int(info)
                if not accepted:
                    cg_failures += 1
                u_new = solution.reshape(crack.shape + (3,))
                if not use_disp_bc:
                    u_new = self._remove_rigid_translation(u_new)
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
            "displacement_bc_x": bool(use_disp_bc),
            "displacement_bc_hard": bool(use_hard_bc),
            "nondim_kinematics": bool(use_nondim_kin),
            "length_ref": float(length_ref),
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
            "preconditioner": str(self.config.preconditioner),
            "solution_clipped": bool(solution_clipped),
        }

        total_strain = sym_grad(u, spacing_solve, self.grid.periodic) + macro_eff
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
        displacement_out = u * displacement_scale if use_nondim_kin else u
        return displacement_out, total_strain, stress


__all__ = ["MechanicalEquilibriumSolver", "MechanicalConfig"]
