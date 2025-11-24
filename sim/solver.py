"""
Alternating solver with mechanical equilibrium and spectral PFC update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .energy import Array, FreeEnergy, PFCCoupling
from .mechanics import MechanicalEquilibriumSolver, MechanicalConfig
from .pfc import PFCEvolver


@dataclass
class SolverConfig:
    dt: float = 1e-2
    plastic_relax: float = 0.1
    crack_relax: float = 0.05  # 稍微调大，让裂纹在有驱动力时能生长
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
    ) -> None:
        self.coupling = coupling
        self.energy = energy
        self.mechanical = mechanical
        self.pfc = pfc
        self.config = config or SolverConfig()
        self.mu_extra_from_stress = mu_extra_from_stress
        self.state: Dict[str, Array] = {}

    def initialize_state(self, orientation_field: Array, seed: int = 0) -> None:
        psi = self.coupling.initialize_density(orientation_field.shape[:-2], seed)
        crack = np.zeros_like(psi)
        plastic = np.zeros_like(psi)
        plastic_vec = np.zeros(psi.shape + (3,))
        displacement = np.zeros(psi.shape + (3,))
        history = np.zeros_like(psi)
        stress = np.zeros(psi.shape + (3, 3))
        stress_vm = np.zeros_like(psi)
        self.state = {
            "psi": psi,
            "crack": crack,
            "plastic": plastic,
            "plastic_vec": plastic_vec,
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
        plastic = self.state["plastic"]
        plastic_vec = self.state["plastic_vec"]
        displacement = self.state["displacement"]
        history = self.state["history"]

        # 1. 力学求解
        displacement, strain, stress = self.mechanical.solve(displacement, crack, macro_strain)
        stress_vm = von_mises(stress)

        # 2. 塑性场更新
        eps_eq, eps_vec = self.coupling.plastic_measures(
            psi,
            plastic,
            plastic_vec,
            strain=strain,
            load_axis=self.config.load_axis,
            mech_weight=self.config.mech_plastic_weight,
        )
        plastic = plastic + self.config.plastic_relax * (eps_eq - plastic)
        plastic_vec = plastic_vec + self.config.plastic_relax * (eps_vec - plastic_vec)

        # 3. 裂纹更新 (使用 l0 修正量纲)
        pos_energy = self.energy.positive_strain_energy(strain, self.mechanical.stiffness)
        # 方向性权重：加载轴上的塑性分量提高历史能量权重
        load_comp = np.clip(plastic_vec[..., self.config.load_axis], 0.0, None)
        anisotropic_boost = 1.0 + self.config.dir_coupling * load_comp
        history = np.maximum(history, pos_energy * anisotropic_boost)
        
        # 获取 l0 和 toughness
        l0 = self.energy.fracture.l0
        toughness = self.coupling.degraded_toughness(psi, plastic)
        
        # 计算驱动力 (引入 l0 进行无量纲化)
        # 能量密度历史 * 长度尺度 / 韧性
        driving_force = (history * l0 / (toughness + 1e-12)) * (1.0 - crack)
        # 同样用方向性塑性增强驱动力（避免负向缩减）
        driving_force = driving_force * anisotropic_boost
        crack = np.clip(crack + self.config.crack_relax * driving_force, 0.0, 1.0)

        # 4. PFC 演化 (【关键】传入宏观应变)
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
                "plastic_vec": plastic_vec,
                "displacement": displacement,
                "history": history,
                "stress": stress,
                "stress_vm": stress_vm,
            }
        )
        total_E = self.energy.total_energy(strain, crack, psi, self.mechanical.stiffness, plastic)
        return total_E


__all__ = ["SolverConfig", "AlternatingSolver"]
