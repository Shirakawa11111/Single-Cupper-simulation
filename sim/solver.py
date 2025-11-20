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
    crack_relax: float = 0.01
    mechanical: MechanicalConfig = field(default_factory=MechanicalConfig)


class AlternatingSolver:
    def __init__(
        self,
        coupling: PFCCoupling,
        energy: FreeEnergy,
        mechanical: MechanicalEquilibriumSolver,
        pfc: PFCEvolver,
        config: SolverConfig | None = None,
    ) -> None:
        self.coupling = coupling
        self.energy = energy
        self.mechanical = mechanical
        self.pfc = pfc
        self.config = config or SolverConfig()
        self.state: Dict[str, Array] = {}

    def initialize_state(self, orientation_field: Array, seed: int = 0) -> None:
        psi = self.coupling.initialize_density(orientation_field.shape[:-2], seed)
        crack = np.zeros_like(psi)
        plastic = np.zeros_like(psi)
        displacement = np.zeros(psi.shape + (3,))
        history = np.zeros_like(psi)
        self.state = {
            "psi": psi,
            "crack": crack,
            "plastic": plastic,
            "displacement": displacement,
            "history": history,
        }

    def step(self, macro_strain: Tuple[float, float, float]) -> float:
        if not self.state:
            raise RuntimeError("initialize_state must be called.")
        psi = self.state["psi"]
        crack = self.state["crack"]
        plastic = self.state["plastic"]
        displacement = self.state["displacement"]
        history = self.state["history"]

        displacement, strain, stress = self.mechanical.solve(displacement, crack, macro_strain)

        eps_eq = self.coupling.equivalent_plastic_strain(psi, plastic)
        plastic = plastic + self.config.plastic_relax * (eps_eq - plastic)

        pos_energy = self.energy.positive_strain_energy(strain, self.mechanical.stiffness)
        history = np.maximum(history, pos_energy)
        toughness = self.coupling.degraded_toughness(psi, plastic)
        # Scale by the phase-field length to compare elastic energy density to critical fracture energy density.
        driving_force = (history * self.energy.fracture.l0 / (toughness + 1e-12)) * (1.0 - crack)
        crack = np.clip(crack + self.config.crack_relax * driving_force, 0.0, 1.0)

        psi = self.pfc.step(psi)
        psi = self.coupling.constraint.project(psi)

        self.state.update(
            {"psi": psi, "crack": crack, "plastic": plastic, "displacement": displacement, "history": history}
        )
        total_E = self.energy.total_energy(strain, crack, psi, self.mechanical.stiffness, plastic)
        return total_E


__all__ = ["SolverConfig", "AlternatingSolver"]
