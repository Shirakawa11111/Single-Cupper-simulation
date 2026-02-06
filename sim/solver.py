"""
Alternating solver with mechanical equilibrium and spectral PFC update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .energy import Array, FreeEnergy, PFCCoupling
from .dislocation import gnd_from_slip
from .mechanics import MechanicalEquilibriumSolver, MechanicalConfig
from .pfc import PFCEvolver


@dataclass
class SolverConfig:
    dt: float = 1e-2
    # Plastic relaxation per step (nd). Combine with PFCCoupling.flow_scale to match σ–ε softening rate.
    plastic_relax: float = 0.12
    # Under-relaxation for crack solve (0 freezes crack, 1 full update)
    crack_relax: float = 1.0  # set 0.0 to freeze crack if needed
    crack_eta: float = 0.0  # viscous regularization η_φ
    crack_tol: float = 1e-6
    crack_max_iters: int = 400
    # Accept crack CG result when relative residual is below threshold.
    crack_accept_rel_residual: float = 5e-3
    # Accept finite crack CG result even when info>0 to avoid crack-update stalls.
    crack_accept_incomplete: bool = True
    # 加载方向（用于方向性塑性耦合），默认沿 x 轴
    load_axis: int = 0
    # 方向性耦合强度：>0 时，加载方向上的塑性分量会增强裂纹驱动力
    dir_coupling: float = 0.3
    # 机械应变对塑性的权重 (0~1)
    mech_plastic_weight: float = 0.7
    # Whether to evolve PFC after initialization
    pfc_active: bool = True
    # Optional GND diagnostics
    gnd_active: bool = False
    gnd_burgers: float = 1.0
    # Fail fast when non-finite values appear
    fail_on_nonfinite: bool = True
    mechanical: MechanicalConfig = field(default_factory=MechanicalConfig)


def von_mises(stress: Array) -> Array:
    """Compute von Mises equivalent stress for a stress tensor field."""
    tr = np.trace(stress, axis1=-2, axis2=-1)[..., None, None]
    dev = stress - tr / 3.0
    vm = np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1)))
    return vm


class AlternatingSolver:
    def __init__(
        self,
        coupling: PFCCoupling,
        energy: FreeEnergy,
        mechanical: MechanicalEquilibriumSolver,
        pfc: PFCEvolver,
        config: SolverConfig | None = None,
        mu_extra_from_stress: callable | None = None,
        grain_mask: Array | None = None,
    ) -> None:
        self.coupling = coupling
        self.energy = energy
        self.mechanical = mechanical
        self.pfc = pfc
        self.config = config or SolverConfig()
        self.mu_extra_from_stress = mu_extra_from_stress
        self.grain_mask = grain_mask
        self.state: Dict[str, Array] = {}
        self.last_step_diagnostics: Dict[str, Any] = {}

    def _check_finite(self, name: str, value: Array) -> int:
        arr = np.asarray(value)
        finite = np.isfinite(arr)
        if bool(np.all(finite)):
            return 0
        bad = int(arr.size - np.count_nonzero(finite))
        if self.config.fail_on_nonfinite:
            finite_vals = arr[finite]
            if finite_vals.size:
                vmin = float(np.min(finite_vals))
                vmax = float(np.max(finite_vals))
                span = f"[{vmin:.3e}, {vmax:.3e}]"
            else:
                span = "[n/a, n/a]"
            raise FloatingPointError(f"Non-finite detected in `{name}`: bad={bad}, finite_range={span}")
        return bad

    def initialize_state(self, orientation_field: Array, seed: int = 0) -> None:
        self.orientation = orientation_field
        psi = self.coupling.initialize_density(orientation_field.shape[:-2], seed)
        crack = np.zeros_like(psi)
        plastic_eq = np.zeros_like(psi)  # accumulated equivalent plastic strain (for toughness degradation)
        plastic_inst = np.zeros_like(psi)  # instantaneous equivalent plastic proxy
        plastic_vec = np.zeros(psi.shape + (3,))  # current directional surrogate (bounded 0-1)
        displacement = np.zeros(psi.shape + (3,))
        history = np.zeros_like(psi)
        stress = np.zeros(psi.shape + (3, 3))
        stress_vm = np.zeros_like(psi)
        plastic_tensor = np.zeros(psi.shape + (3, 3))  # accumulated plastic tensor (eigenstrain)
        accum_plastic = plastic_eq.copy()  # track accumulated scalar for hardening/toughness
        backstress = np.zeros_like(stress)  # legacy placeholder (no longer used)
        n_slip = len(self.coupling.slip_systems)
        gamma_s = np.zeros((n_slip,) + psi.shape)
        chi_s = np.zeros_like(gamma_s)
        tau_c = np.full_like(psi, self.coupling.yield_tau)
        gnd_density = np.zeros_like(psi)
        self.state = {
            "psi": psi,
            "crack": crack,
            "plastic": plastic_eq,
            "plastic_inst": plastic_inst,
            "plastic_vec": plastic_vec,
            "plastic_tensor": plastic_tensor,
            "accum_plastic": accum_plastic,
            "backstress": backstress,
            "gamma_s": gamma_s,
            "chi_s": chi_s,
            "tau_c": tau_c,
            "gnd_density": gnd_density,
            "displacement": displacement,
            "history": history,
            "stress": stress,
            "stress_vm": stress_vm,
        }

    def step(self, macro_strain: Tuple[float, float, float]) -> float:
        if not self.state:
            raise RuntimeError("initialize_state must be called.")
        psi = self.state["psi"]
        crack = self.state["crack"]
        plastic = self.state["plastic"]  # accumulated equivalent plastic strain
        plastic_vec = self.state["plastic_vec"]
        plastic_tensor = self.state.get("plastic_tensor", np.zeros(psi.shape + (3, 3)))  # accumulated tensor
        accum_plastic = self.state.get("accum_plastic", plastic.copy())
        backstress = self.state.get("backstress", np.zeros(psi.shape + (3, 3)))
        gamma_s = self.state.get("gamma_s")
        chi_s = self.state.get("chi_s")
        gnd_density = self.state.get("gnd_density")
        displacement = self.state["displacement"]
        history = self.state["history"]
        nonfinite_count = 0

        # 1. 力学求解
        displacement, strain, stress = self.mechanical.solve(displacement, crack, macro_strain, plastic_strain=plastic_tensor)
        mech_info_first = dict(getattr(self.mechanical, "last_solve_info", {}))
        nonfinite_count += self._check_finite("displacement_after_mech_1", displacement)
        nonfinite_count += self._check_finite("strain_after_mech_1", strain)
        nonfinite_count += self._check_finite("stress_after_mech_1", stress)
        stress_vm = von_mises(stress)
        nonfinite_count += self._check_finite("stress_vm_after_mech_1", stress_vm)

        # 2. 塑性场更新
        if gamma_s is None or chi_s is None:
            raise RuntimeError("gamma_s/chi_s not initialized; call initialize_state first.")
        need_gnd_hardening = self.coupling.h_gnd > 0.0
        if need_gnd_hardening:
            gnd_density, _ = gnd_from_slip(
                gamma_s,
                self.coupling.slip_systems,
                self.orientation,
                self.mechanical.grid,
                burgers=self.config.gnd_burgers,
            )
        gamma_dot, epsp_inc, eps_vec, eps_eq, tau_c = self.coupling.slip_system_flow(
            stress,
            gamma_s,
            chi_s,
            dt=self.config.dt,
            orientation=self.orientation,
            gnd_density=gnd_density if need_gnd_hardening else None,
            dev_only=True,
        )
        relax = self.config.plastic_relax
        gamma_s = gamma_s + relax * self.config.dt * gamma_dot
        chi_s = chi_s + relax * self.config.dt * (
            self.coupling.kin_c * gamma_dot - self.coupling.kin_d * np.abs(gamma_dot) * chi_s
        )
        epsp_increment = relax * epsp_inc
        plastic_tensor = plastic_tensor + epsp_increment
        accum_plastic = np.sum(np.abs(gamma_s), axis=0)
        plastic = accum_plastic
        plastic_inst = eps_eq * relax
        plastic_vec = np.clip(plastic_vec + relax * (eps_vec - plastic_vec), 0.0, 1.0)
        backstress = backstress * 0.0
        nonfinite_count += self._check_finite("gamma_s", gamma_s)
        nonfinite_count += self._check_finite("chi_s", chi_s)
        nonfinite_count += self._check_finite("plastic_tensor", plastic_tensor)
        nonfinite_count += self._check_finite("accum_plastic", accum_plastic)

        if self.config.gnd_active and not need_gnd_hardening:
            gnd_density, _ = gnd_from_slip(
                gamma_s,
                self.coupling.slip_systems,
                self.orientation,
                self.mechanical.grid,
                burgers=self.config.gnd_burgers,
            )
        # Re-evaluate displacement/strain/stress with updated plastic tensor (one inner iteration)
        displacement, strain, stress = self.mechanical.solve(displacement, crack, macro_strain, plastic_strain=plastic_tensor)
        mech_info_second = dict(getattr(self.mechanical, "last_solve_info", {}))
        nonfinite_count += self._check_finite("displacement_after_mech_2", displacement)
        nonfinite_count += self._check_finite("strain_after_mech_2", strain)
        nonfinite_count += self._check_finite("stress_after_mech_2", stress)
        stress_vm = von_mises(stress)
        nonfinite_count += self._check_finite("stress_vm_after_mech_2", stress_vm)

        # 3. 裂纹驱动力: 使用弹性应变 (去除塑性) 的正能量部分
        strain_el = strain - plastic_tensor
        pos_energy = self.energy.positive_strain_energy(strain_el, self.mechanical.stiffness)
        # 方向性权重：加载轴上的塑性分量提高历史能量权重
        load_comp = np.clip(plastic_vec[..., self.config.load_axis], 0.0, None)
        anisotropic_boost = 1.0 + self.config.dir_coupling * load_comp
        history = np.maximum(history, pos_energy * anisotropic_boost)

        # 4. AT2 裂纹求解（SPD 线性方程，带可选粘性与欠松弛）
        l0 = self.energy.fracture.l0
        toughness = self.coupling.degraded_toughness(psi, accum_plastic, grain_mask=self.grain_mask)
        eta_dt = self.config.crack_eta / max(self.config.dt, 1e-12)
        coeff_diag = (toughness / l0) + 2.0 * history + eta_dt

        grid = self.mechanical.grid
        spacing = self.mechanical.spacing

        def apply_crack_operator(vec: np.ndarray) -> np.ndarray:
            phi = vec.reshape(crack.shape)
            out = coeff_diag * phi
            for ax, dx in enumerate(spacing):
                phi_f = np.roll(phi, -1, axis=ax)
                t_f = 0.5 * (toughness + np.roll(toughness, -1, axis=ax))
                if not grid.periodic[ax]:
                    sl_last = [slice(None)] * phi.ndim
                    sl_last[ax] = -1
                    phi_f[tuple(sl_last)] = phi[tuple(sl_last)]
                    t_f[tuple(sl_last)] = toughness[tuple(sl_last)]
                grad_f = (phi_f - phi) / dx
                flux_f = t_f * l0 * grad_f
                if not grid.periodic[ax]:
                    sl_last = [slice(None)] * phi.ndim
                    sl_last[ax] = -1
                    flux_f[tuple(sl_last)] = 0.0
                flux_b = np.roll(flux_f, 1, axis=ax)
                if not grid.periodic[ax]:
                    sl_first = [slice(None)] * phi.ndim
                    sl_first[ax] = 0
                    flux_b[tuple(sl_first)] = 0.0
                out += (flux_f - flux_b) / dx
            return out.reshape(-1)

        rhs = (2.0 * history + eta_dt * crack).reshape(-1)
        linop = LinearOperator((crack.size, crack.size), matvec=apply_crack_operator)
        phi0 = crack.reshape(-1)
        phi_vec, info = cg(linop, rhs, x0=phi0, rtol=self.config.crack_tol, atol=0.0, maxiter=self.config.crack_max_iters)
        crack_cg_info = int(info)
        phi_finite = bool(np.all(np.isfinite(phi_vec)))
        crack_cg_rel_res = float("inf")
        if phi_finite:
            res = linop.matvec(phi_vec) - rhs
            crack_cg_rel_res = float(np.linalg.norm(res) / (np.linalg.norm(rhs) + 1e-16))
        crack_cg_accepted = bool(
            phi_finite
            and (
                crack_cg_info == 0
                or crack_cg_rel_res <= self.config.crack_accept_rel_residual
                or (self.config.crack_accept_incomplete and crack_cg_info > 0)
            )
        )
        phi_new = phi_vec.reshape(crack.shape) if crack_cg_accepted else crack
        # Irreversibility and bounds
        phi_new = np.maximum(phi_new, crack)
        phi_new = np.clip(phi_new, 0.0, 1.0)
        omega = np.clip(self.config.crack_relax, 0.0, 1.0)
        crack = (1.0 - omega) * crack + omega * phi_new
        nonfinite_count += self._check_finite("crack_after_update", crack)

        # 5. PFC 演化 (【关键】传入宏观应变)
        if self.config.pfc_active:
            self.pfc.update_strain(macro_strain)
            # 应力耦合的化学势附加项
            if self.mu_extra_from_stress is not None:
                extra_mu = self.mu_extra_from_stress(stress_vm)
                self.pfc.extra_mu = lambda psi: extra_mu
            psi = self.pfc.step(psi)
            psi = self.coupling.constraint.project(psi)
        nonfinite_count += self._check_finite("psi_after_step", psi)

        self.state.update(
            {
                "psi": psi,
                "crack": crack,
                "plastic": plastic,
                "plastic_inst": plastic_inst,
                "plastic_vec": plastic_vec,
                "plastic_tensor": plastic_tensor,
                "accum_plastic": accum_plastic,
                "backstress": backstress,
                "gamma_s": gamma_s,
                "chi_s": chi_s,
                "tau_c": tau_c,
                "gnd_density": gnd_density,
                "displacement": displacement,
                "history": history,
                "stress": stress,
                "stress_vm": stress_vm,
            }
        )
        mech_failures = int(mech_info_first.get("cg_failures", 0)) + int(mech_info_second.get("cg_failures", 0))
        mech_first_info = int(mech_info_first.get("last_cg_info", 0))
        mech_second_info = int(mech_info_second.get("last_cg_info", 0))
        mech_warning_count = int(mech_info_first.get("runtime_warning_count", 0)) + int(
            mech_info_second.get("runtime_warning_count", 0)
        )
        self.last_step_diagnostics = {
            "mechanical_cg_failures": mech_failures,
            "mechanical_first_cg_info": mech_first_info,
            "mechanical_second_cg_info": mech_second_info,
            "mechanical_last_cg_info": mech_second_info,
            "mechanical_outer_converged": bool(
                mech_info_first.get("outer_converged", True) and mech_info_second.get("outer_converged", True)
            ),
            "mechanical_runtime_warning_count": mech_warning_count,
            "mechanical_gmres_fallback_used": bool(
                mech_info_first.get("gmres_fallback_used", False) or mech_info_second.get("gmres_fallback_used", False)
            ),
            "mechanical_last_solver_used": str(mech_info_second.get("solver_used", mech_info_first.get("solver_used", "cg"))),
            "mechanical_last_rel_residual": float(
                mech_info_second.get("rel_residual", mech_info_first.get("rel_residual", 0.0))
            ),
            "mechanical_last_accepted": bool(mech_info_second.get("accepted", mech_info_first.get("accepted", True))),
            "mechanical_solution_clipped": bool(
                mech_info_second.get("solution_clipped", mech_info_first.get("solution_clipped", False))
            ),
            "crack_cg_info": crack_cg_info,
            "crack_cg_converged": crack_cg_info == 0,
            "crack_cg_rel_residual": crack_cg_rel_res,
            "crack_cg_accepted": crack_cg_accepted,
            "nonfinite_count": nonfinite_count,
            "max_abs_stress_vm": float(np.max(np.abs(stress_vm))),
            "max_crack": float(np.max(crack)),
            "max_abs_displacement": float(np.max(np.abs(displacement))),
        }
        total_E = self.energy.total_energy(
            strain,
            crack,
            psi,
            self.mechanical.stiffness,
            accum_plastic,
            grain_mask=self.grain_mask,
            plastic_tensor=plastic_tensor,
            spacing=self.mechanical.spacing,
            periodic=self.mechanical.grid.periodic,
        )
        return total_E


__all__ = ["SolverConfig", "AlternatingSolver"]
