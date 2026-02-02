"""
Alternating solver with mechanical equilibrium and spectral PFC update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .energy import Array, FreeEnergy, PFCCoupling
from .mechanics import MechanicalEquilibriumSolver, MechanicalConfig
from .pfc import PFCEvolver


@dataclass
class SolverConfig:
    dt: float = 1e-2
    # Plastic relaxation per step (nd). Combine with PFCCoupling.flow_scale to match σ–ε softening rate.
    plastic_relax: float = 0.12
    # Under-relaxation for crack solve (0 freezes crack, 1 full update)
    crack_relax: float = 1.0
    crack_eta: float = 0.0  # viscous regularization η_φ
    crack_tol: float = 1e-6
    crack_max_iters: int = 400
    # 加载方向（用于方向性塑性耦合），默认沿 x 轴
    load_axis: int = 0
    # 方向性耦合强度：>0 时，加载方向上的塑性分量会增强裂纹驱动力
    dir_coupling: float = 0.3
    # 机械应变对塑性的权重 (0~1)
    mech_plastic_weight: float = 0.7
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

    def initialize_state(self, orientation_field: Array, seed: int = 0) -> None:
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
        backstress = np.zeros_like(stress)  # kinematic hardening back-stress tensor
        self.state = {
            "psi": psi,
            "crack": crack,
            "plastic": plastic_eq,
            "plastic_inst": plastic_inst,
            "plastic_vec": plastic_vec,
            "plastic_tensor": plastic_tensor,
            "accum_plastic": accum_plastic,
            "backstress": backstress,
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
        displacement = self.state["displacement"]
        history = self.state["history"]

        # 1. 力学求解
        displacement, strain, stress = self.mechanical.solve(displacement, crack, macro_strain, plastic_strain=plastic_tensor)
        stress_vm = von_mises(stress)

        # 2. 塑性场更新
        eps_eq, eps_vec, epsp_tensor = self.coupling.plastic_measures(
            psi,
            accum_plastic,
            plastic_vec,
            strain=strain,
            stress=stress,
            backstress=backstress,
            load_axis=self.config.load_axis,
            mech_weight=self.config.mech_plastic_weight,
        )
        epsp_increment = self.config.plastic_relax * epsp_tensor
        plastic_tensor = plastic_tensor + epsp_increment
        eq_inc = np.sqrt((2.0 / 3.0) * np.sum(epsp_increment * epsp_increment, axis=(-2, -1)))
        accum_plastic = accum_plastic + eq_inc
        plastic = accum_plastic
        plastic_inst = eps_eq
        plastic_vec = np.clip(plastic_vec + self.config.plastic_relax * (eps_vec - plastic_vec), 0.0, 1.0)
        # kinematic hardening (back-stress): simple Prager/Armstrong-Frederick form
        backstress = backstress + self.coupling.kin_c * epsp_increment - self.coupling.kin_d * eq_inc[..., None, None] * backstress
        backstress = 0.5 * (backstress + np.swapaxes(backstress, -1, -2))  # enforce symmetry
        # Re-evaluate displacement/strain/stress with updated plastic tensor (one inner iteration)
        displacement, strain, stress = self.mechanical.solve(displacement, crack, macro_strain, plastic_strain=plastic_tensor)
        stress_vm = von_mises(stress)

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
        phi_new = crack if info != 0 else phi_vec.reshape(crack.shape)
        # Irreversibility and bounds
        phi_new = np.maximum(phi_new, crack)
        phi_new = np.clip(phi_new, 0.0, 1.0)
        omega = np.clip(self.config.crack_relax, 0.0, 1.0)
        crack = (1.0 - omega) * crack + omega * phi_new

        # 5. PFC 演化 (【关键】传入宏观应变)
        self.pfc.update_strain(macro_strain)
        # 应力耦合的化学势附加项
        if self.mu_extra_from_stress is not None:
            extra_mu = self.mu_extra_from_stress(stress_vm)
            self.pfc.extra_mu = lambda psi: extra_mu
        psi = self.pfc.step(psi)
        psi = self.coupling.constraint.project(psi)

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
                "displacement": displacement,
                "history": history,
                "stress": stress,
                "stress_vm": stress_vm,
            }
        )
        total_E = self.energy.total_energy(
            strain, crack, psi, self.mechanical.stiffness, accum_plastic, grain_mask=self.grain_mask, plastic_tensor=plastic_tensor
        )
        return total_E


__all__ = ["SolverConfig", "AlternatingSolver"]
