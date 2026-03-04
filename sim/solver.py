"""
Alternating solver with mechanical equilibrium and spectral PFC update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .energy import Array, FreeEnergy, PFCCoupling, crack_driving_force
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
    # Optional history localization trigger based on crack level.
    # <=0 disables this feature.
    history_localization_crack_threshold: float = 0.0
    # Multiplier applied to history update when crack >= threshold.
    history_localization_boost: float = 1.0
    # Multiplier applied to history update outside the localized region.
    history_background_scale: float = 1.0
    # Mechanics-plastic fixed-point coupling iterations per load increment.
    coupling_inner_iters: int = 3
    # Minimum inner iterations before convergence check can stop early.
    coupling_min_iters: int = 1
    # Relative tolerance for plastic-tensor increment convergence.
    coupling_tol: float = 5e-4
    # Relative tolerance for stress convergence in the inner loop.
    coupling_stress_tol: float = 5e-4
    # Re-solve mechanics after crack update to remove one-step crack lag.
    post_crack_mech_correction: bool = True
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
        strain = np.zeros(psi.shape + (3, 3))
        history = np.zeros_like(psi)
        stress = np.zeros(psi.shape + (3, 3))
        stress_vm = np.zeros_like(psi)
        plastic_tensor = np.zeros(psi.shape + (3, 3))  # accumulated plastic tensor (eigenstrain)
        accum_plastic = plastic_eq.copy()  # track accumulated scalar for hardening/toughness
        backstress = np.zeros_like(stress)  # effective tensor form reconstructed from slip back-stresses
        n_slip = len(self.coupling.slip_systems)
        gamma_s = np.zeros((n_slip,) + psi.shape)
        chi_s = np.zeros_like(gamma_s)
        chi_s2 = np.zeros_like(gamma_s)
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
            "chi_s2": chi_s2,
            "tau_c": tau_c,
            "gnd_density": gnd_density,
            "displacement": displacement,
            "strain": strain,
            "history": history,
            "stress": stress,
            "stress_vm": stress_vm,
        }

    def compute_energy_fields(self) -> Dict[str, Array]:
        if not self.state:
            raise RuntimeError("initialize_state must be called.")
        strain = self.state.get("strain")
        if strain is None:
            raise RuntimeError("strain field not found in solver state.")
        crack = self.state["crack"]
        psi = self.state["psi"]
        accum_plastic = self.state.get("accum_plastic", self.state["plastic"])
        plastic_tensor = self.state.get("plastic_tensor")
        comps = self.energy.energy_density_components(
            strain,
            crack,
            psi,
            self.mechanical.stiffness,
            plastic_eq=accum_plastic,
            grain_mask=self.grain_mask,
            plastic_tensor=plastic_tensor,
        )
        drive = crack_driving_force(
            crack,
            comps["toughness"],
            self.energy.fracture.l0,
            spacing=self.mechanical.spacing,
            periodic=self.mechanical.grid.periodic,
        )
        fields = {
            "energy_elastic": comps["elastic"],
            "energy_pfc": comps["pfc"],
            "energy_crack": comps["crack"],
            "energy_total_density": comps["total"],
            "toughness": comps["toughness"],
            "crack_driving_force": np.nan_to_num(drive),
        }
        self.state.update(fields)
        return fields

    def _assemble_backstress_tensor(self, chi_s: Array, chi_s2: Array | None = None) -> Array:
        """
        Build an effective back-stress tensor field from slip back-stress states.

        This tensor is used for diagnostics/visualization; slip-level return mapping
        still uses chi_s directly in `PFCCoupling.slip_system_flow`.
        """
        backstress = np.zeros(chi_s.shape[1:] + (3, 3), dtype=float)
        orientation = self.orientation
        for k, (m, n) in enumerate(self.coupling.slip_systems):
            if orientation.shape == (3, 3):
                m_lab = np.einsum("ij,j->i", orientation, m, optimize=True)
                n_lab = np.einsum("ij,j->i", orientation, n, optimize=True)
                mn = np.outer(m_lab, n_lab)
            else:
                m_lab = np.einsum("...ij,j->...i", orientation, m, optimize=True)
                n_lab = np.einsum("...ij,j->...i", orientation, n, optimize=True)
                mn = np.einsum("...i,...j->...ij", m_lab, n_lab, optimize=True)
            sym_mn = 0.5 * (mn + np.swapaxes(mn, -1, -2))
            chi_total = chi_s[k] if chi_s2 is None else (chi_s[k] + chi_s2[k])
            backstress += chi_total[..., None, None] * sym_mn
        return backstress

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
        chi_s2 = self.state.get("chi_s2")
        gnd_density = self.state.get("gnd_density")
        displacement = self.state["displacement"]
        history = self.state["history"]
        # Keep previous committed state. If mechanics rejects this increment, do not commit nonlinear updates.
        prev_psi = self.state["psi"]
        prev_crack = self.state["crack"]
        prev_plastic = self.state["plastic"]
        prev_plastic_inst = self.state.get("plastic_inst", np.zeros_like(psi))
        prev_plastic_vec = self.state["plastic_vec"]
        prev_plastic_tensor = self.state.get("plastic_tensor", np.zeros(psi.shape + (3, 3)))
        prev_accum_plastic = self.state.get("accum_plastic", self.state["plastic"])
        prev_backstress = self.state.get("backstress", np.zeros(psi.shape + (3, 3)))
        prev_gamma_s = self.state.get("gamma_s")
        prev_chi_s = self.state.get("chi_s")
        prev_chi_s2 = self.state.get("chi_s2")
        prev_tau_c = self.state.get("tau_c", np.full_like(psi, self.coupling.yield_tau))
        prev_gnd_density = self.state.get("gnd_density", np.zeros_like(psi))
        prev_displacement = self.state["displacement"]
        prev_strain = self.state.get("strain", np.zeros(psi.shape + (3, 3)))
        prev_history = self.state.get("history", np.zeros_like(psi))
        prev_stress = self.state.get("stress", np.zeros(psi.shape + (3, 3)))
        prev_stress_vm = self.state.get("stress_vm", np.zeros_like(psi))
        nonfinite_count = 0
        if gamma_s is None or chi_s is None:
            raise RuntimeError("gamma_s/chi_s not initialized; call initialize_state first.")
        if chi_s2 is None:
            chi_s2 = np.zeros_like(chi_s)
        # 1. 力学-塑性内耦合迭代（替代单次回代）
        coupling_iters = max(1, int(self.config.coupling_inner_iters))
        coupling_min_iters = max(1, min(coupling_iters, int(self.config.coupling_min_iters)))
        dt_sub = self.config.dt / coupling_iters
        relax = self.config.plastic_relax
        need_gnd_hardening = self.coupling.h_gnd > 0.0
        mech_infos: list[dict[str, Any]] = []
        prev_stress = None
        coupling_converged = False
        coupling_residual = float("inf")
        coupling_stress_residual = float("inf")
        tau_c = self.state.get("tau_c", np.full_like(psi, self.coupling.yield_tau))
        plastic_inst = self.state.get("plastic_inst", np.zeros_like(psi))

        for inner in range(coupling_iters):
            displacement, strain, stress = self.mechanical.solve(
                displacement,
                crack,
                macro_strain,
                plastic_strain=plastic_tensor,
            )
            mech_infos.append(dict(getattr(self.mechanical, "last_solve_info", {})))
            nonfinite_count += self._check_finite(f"displacement_after_mech_inner_{inner+1}", displacement)
            nonfinite_count += self._check_finite(f"strain_after_mech_inner_{inner+1}", strain)
            nonfinite_count += self._check_finite(f"stress_after_mech_inner_{inner+1}", stress)
            stress_vm = von_mises(stress)
            nonfinite_count += self._check_finite(f"stress_vm_after_mech_inner_{inner+1}", stress_vm)
            mech_inner_accepted = bool(mech_infos[-1].get("accepted", True))
            if not mech_inner_accepted:
                coupling_converged = False
                coupling_residual = float("inf")
                coupling_stress_residual = float("inf")
                break

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
                chi_s2=chi_s2,
                dt=dt_sub,
                orientation=self.orientation,
                gnd_density=gnd_density if need_gnd_hardening else None,
                dev_only=True,
            )
            gamma_s_new = gamma_s + relax * dt_sub * gamma_dot
            chi_s_new = chi_s + relax * dt_sub * (
                self.coupling.kin_c * gamma_dot - self.coupling.kin_d * np.abs(gamma_dot) * chi_s
            )
            chi_s2_new = chi_s2 + relax * dt_sub * (
                self.coupling.kin_c2 * gamma_dot - self.coupling.kin_d2 * np.abs(gamma_dot) * chi_s2
            )
            epsp_increment = relax * epsp_inc
            plastic_tensor_new = plastic_tensor + epsp_increment
            accum_plastic_new = np.sum(np.abs(gamma_s_new), axis=0)
            plastic_new = accum_plastic_new
            plastic_inst_new = eps_eq * relax
            plastic_vec_new = np.clip(plastic_vec + relax * (eps_vec - plastic_vec), 0.0, 1.0)

            d_plastic = plastic_tensor_new - plastic_tensor
            d_plastic_num = float(np.linalg.norm(d_plastic.reshape(-1)))
            d_plastic_den = max(
                float(np.linalg.norm(plastic_tensor_new.reshape(-1))),
                float(np.linalg.norm(plastic_tensor.reshape(-1))),
                1e-10,
            )
            coupling_residual = float(
                d_plastic_num / d_plastic_den
            )
            if prev_stress is None:
                coupling_stress_residual = float("inf")
            else:
                d_stress_num = float(np.linalg.norm((stress - prev_stress).reshape(-1)))
                d_stress_den = max(
                    float(np.linalg.norm(stress.reshape(-1))),
                    float(np.linalg.norm(prev_stress.reshape(-1))),
                    1e-10,
                )
                coupling_stress_residual = float(
                    d_stress_num / d_stress_den
                )

            gamma_s = gamma_s_new
            chi_s = chi_s_new
            chi_s2 = chi_s2_new
            plastic_tensor = plastic_tensor_new
            accum_plastic = accum_plastic_new
            plastic = plastic_new
            plastic_inst = plastic_inst_new
            plastic_vec = plastic_vec_new
            prev_stress = stress.copy()

            nonfinite_count += self._check_finite("gamma_s", gamma_s)
            nonfinite_count += self._check_finite("chi_s", chi_s)
            nonfinite_count += self._check_finite("chi_s2", chi_s2)
            nonfinite_count += self._check_finite("plastic_tensor", plastic_tensor)
            nonfinite_count += self._check_finite("accum_plastic", accum_plastic)

            if (
                inner + 1 >= coupling_min_iters
                and coupling_residual <= self.config.coupling_tol
                and coupling_stress_residual <= self.config.coupling_stress_tol
            ):
                coupling_converged = True
                break

        # 2. 与更新后的塑性场一致的最终力学求解
        displacement, strain, stress = self.mechanical.solve(
            displacement,
            crack,
            macro_strain,
            plastic_strain=plastic_tensor,
        )
        mech_infos.append(dict(getattr(self.mechanical, "last_solve_info", {})))
        nonfinite_count += self._check_finite("displacement_after_mech_final", displacement)
        nonfinite_count += self._check_finite("strain_after_mech_final", strain)
        nonfinite_count += self._check_finite("stress_after_mech_final", stress)
        stress_vm = von_mises(stress)
        nonfinite_count += self._check_finite("stress_vm_after_mech_final", stress_vm)

        if self.config.gnd_active or need_gnd_hardening:
            gnd_density, _ = gnd_from_slip(
                gamma_s,
                self.coupling.slip_systems,
                self.orientation,
                self.mechanical.grid,
                burgers=self.config.gnd_burgers,
            )
        backstress = self._assemble_backstress_tensor(chi_s, chi_s2)
        nonfinite_count += self._check_finite("backstress_tensor", backstress)

        mech_last_info_obj = mech_infos[-1] if mech_infos else {}
        mechanical_step_accepted = bool(mech_last_info_obj.get("accepted", True))
        state_committed = bool(mechanical_step_accepted)
        post_crack_mech_done = False
        crack_cg_info = -1
        crack_cg_rel_res = float("inf")
        crack_cg_accepted = False
        stress_vm_diag = stress_vm
        displacement_diag = displacement

        if state_committed:
            # 3. 裂纹驱动力: 使用弹性应变 (去除塑性) 的正能量部分
            strain_el = strain - plastic_tensor
            pos_energy = self.energy.positive_strain_energy(strain_el, self.mechanical.stiffness)
            # 方向性权重：加载轴上的塑性分量提高历史能量权重
            load_comp = np.clip(plastic_vec[..., self.config.load_axis], 0.0, None)
            anisotropic_boost = 1.0 + self.config.dir_coupling * load_comp
            history_candidate = pos_energy * anisotropic_boost
            loc_thr = float(self.config.history_localization_crack_threshold)
            if loc_thr > 0.0:
                loc_boost = max(float(self.config.history_localization_boost), 0.0)
                bg_scale = max(float(self.config.history_background_scale), 0.0)
                if (loc_boost != 1.0) or (bg_scale != 1.0):
                    loc_mask = crack >= loc_thr
                    loc_weight = np.where(loc_mask, loc_boost, bg_scale)
                    history_candidate = history_candidate * loc_weight
            history = np.maximum(history, history_candidate)

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
            phi_vec, info = cg(
                linop,
                rhs,
                x0=phi0,
                rtol=self.config.crack_tol,
                atol=0.0,
                maxiter=self.config.crack_max_iters,
            )
            crack_cg_info = int(info)
            phi_finite = bool(np.all(np.isfinite(phi_vec)))
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

            # 5. 裂纹更新后力学校正，去除一步滞后误差
            if self.config.post_crack_mech_correction:
                displacement, strain, stress = self.mechanical.solve(
                    displacement,
                    crack,
                    macro_strain,
                    plastic_strain=plastic_tensor,
                )
                mech_infos.append(dict(getattr(self.mechanical, "last_solve_info", {})))
                nonfinite_count += self._check_finite("displacement_after_crack_correction", displacement)
                nonfinite_count += self._check_finite("strain_after_crack_correction", strain)
                nonfinite_count += self._check_finite("stress_after_crack_correction", stress)
                stress_vm = von_mises(stress)
                nonfinite_count += self._check_finite("stress_vm_after_crack_correction", stress_vm)
                stress_vm_diag = stress_vm
                displacement_diag = displacement
                post_crack_mech_done = True

            # 6. PFC 演化 (【关键】传入宏观应变)
            if self.config.pfc_active:
                self.pfc.update_strain(macro_strain)
                # 应力耦合的化学势附加项
                if self.mu_extra_from_stress is not None:
                    extra_mu = self.mu_extra_from_stress(stress_vm)
                    self.pfc.extra_mu = lambda psi: extra_mu
                psi = self.pfc.step(psi)
                psi = self.coupling.constraint.project(psi)
            nonfinite_count += self._check_finite("psi_after_step", psi)
        else:
            # Reject this increment and keep previously committed fields.
            coupling_converged = False
            psi = prev_psi
            crack = prev_crack
            plastic = prev_plastic
            plastic_inst = prev_plastic_inst
            plastic_vec = prev_plastic_vec
            plastic_tensor = prev_plastic_tensor
            accum_plastic = prev_accum_plastic
            backstress = prev_backstress
            gamma_s = prev_gamma_s
            chi_s = prev_chi_s
            chi_s2 = prev_chi_s2
            tau_c = prev_tau_c
            gnd_density = prev_gnd_density
            displacement = prev_displacement
            strain = prev_strain
            history = prev_history
            stress = prev_stress
            stress_vm = prev_stress_vm

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
                "chi_s2": chi_s2,
                "tau_c": tau_c,
                "gnd_density": gnd_density,
                "displacement": displacement,
                "strain": strain,
                "history": history,
                "stress": stress,
                "stress_vm": stress_vm,
            }
        )
        mech_failures = int(sum(int(info.get("cg_failures", 0)) for info in mech_infos))
        mech_warning_count = int(sum(int(info.get("runtime_warning_count", 0)) for info in mech_infos))
        mech_first_info_obj = mech_infos[0] if mech_infos else {}
        mech_second_info_obj = mech_infos[1] if len(mech_infos) > 1 else mech_first_info_obj
        mech_last_info_obj = mech_infos[-1] if mech_infos else {}
        self.last_step_diagnostics = {
            "mechanical_cg_failures": mech_failures,
            "mechanical_first_cg_info": int(mech_first_info_obj.get("last_cg_info", 0)),
            "mechanical_second_cg_info": int(mech_second_info_obj.get("last_cg_info", 0)),
            "mechanical_last_cg_info": int(mech_last_info_obj.get("last_cg_info", 0)),
            "mechanical_outer_converged": bool(
                all(bool(info.get("outer_converged", True)) for info in mech_infos)
            ),
            "mechanical_runtime_warning_count": mech_warning_count,
            "mechanical_gmres_fallback_used": bool(
                any(bool(info.get("gmres_fallback_used", False)) for info in mech_infos)
            ),
            "mechanical_last_solver_used": str(mech_last_info_obj.get("solver_used", "cg")),
            "mechanical_last_rel_residual": float(
                mech_last_info_obj.get("rel_residual", 0.0)
            ),
            "mechanical_last_accepted": bool(mech_last_info_obj.get("accepted", True)),
            "mechanical_step_state_committed": bool(state_committed),
            "mechanical_step_rejected": bool(not state_committed),
            "mechanical_solution_clipped": bool(
                any(bool(info.get("solution_clipped", False)) for info in mech_infos)
            ),
            "coupling_inner_iterations_used": int(len(mech_infos) - 1 - int(post_crack_mech_done)),
            "coupling_converged": bool(coupling_converged),
            "coupling_plastic_residual": float(coupling_residual),
            "coupling_stress_residual": float(coupling_stress_residual),
            "post_crack_mech_correction_used": bool(post_crack_mech_done),
            "crack_cg_info": crack_cg_info,
            "crack_cg_converged": crack_cg_info == 0,
            "crack_cg_rel_residual": crack_cg_rel_res,
            "crack_cg_accepted": crack_cg_accepted,
            "crack_update_skipped_due_mech_reject": bool(not state_committed),
            "nonfinite_count": nonfinite_count,
            "max_abs_stress_vm": float(np.max(np.abs(stress_vm_diag))),
            "max_crack": float(np.max(crack)),
            "max_abs_displacement": float(np.max(np.abs(displacement_diag))),
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
