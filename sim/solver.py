"""
Alternating solver for coupled PFC–phase-field evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .energy import Array, FreeEnergy, PFCCoupling
from .operators import GridSpec, HybridElasticOperator


@dataclass
class SolverConfig:
    dt: float = 1e-2
    plastic_relax: float = 0.1
    crack_relax: float = 0.05
    max_iters: int = 50
    tol: float = 1e-6


class AlternatingSolver:
    """
    Implements the sequential elastic → plastic → crack → PFC update loop.
    """

    def __init__(
        self,
        grid: GridSpec,
        energy: FreeEnergy,
        elastic_operator: HybridElasticOperator,
        coupling: PFCCoupling,
        config: SolverConfig | None = None,
    ) -> None:
        self.grid = grid
        self.energy = energy
        self.elastic = elastic_operator
        self.coupling = coupling
        self.config = config or SolverConfig()
        self.state: Dict[str, Array] = {}

    def initialize_state(self, seed: int = 0) -> None:
        psi = self.coupling.initialize_density(self.grid.shape, seed)
        crack = np.zeros(self.grid.shape)
        plastic = np.zeros(self.grid.shape)
        strain = np.zeros(self.grid.shape + (len(self.grid.shape), len(self.grid.shape)))
        self.state = {"psi": psi, "crack": crack, "plastic": plastic, "strain": strain}

    def step(self, target_strain: Tuple[float, float, float]) -> float:
        if not self.state:
            raise RuntimeError("Call initialize_state first.")

        psi = self.state["psi"]
        crack = self.state["crack"]
        plastic = self.state["plastic"]
        strain = self.state["strain"]

        # Elastic update: impose macroscopic diagonal strain.
        diag = np.array(target_strain)
        strain = np.zeros_like(strain)
        for i in range(len(diag)):
            strain[..., i, i] = diag[i]

        # Plastic update: simple relaxation towards equivalent strain.
        eps_eq = self.coupling.equivalent_plastic_strain(psi, plastic)
        plastic += self.config.plastic_relax * (eps_eq - plastic)

        # Crack update: gradient descent on local energy derivative.
        toughness = self.coupling.degraded_toughness(psi, plastic)
        grad_crack = crack - self.config.crack_relax * (toughness * crack)
        crack = np.clip(grad_crack, 0.0, 1.0)

        # PFC update: conserved dynamics (simple Laplacian smoothing).
        lap = sum(
            np.gradient(np.gradient(psi, self.grid.spacing[i], axis=i), self.grid.spacing[i], axis=i)
            for i in range(len(self.grid.shape))
        )
        psi += self.config.dt * (self.coupling.pfc_params.r * psi - self.coupling.pfc_params.u * psi**3 + lap)
        psi = np.clip(psi, -1.0, 1.0)
        psi = self.coupling.constraint.project(psi)

        self.state.update({"psi": psi, "crack": crack, "plastic": plastic, "strain": strain})
        return self.energy.total_energy(strain, crack, psi, plastic)


__all__ = ["SolverConfig", "AlternatingSolver"]
