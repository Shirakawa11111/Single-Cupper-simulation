"""
Energy models for coupling a phase-field fracture solver with a
Phase-Field Crystal (PFC) density field.

The design follows the ductile-fracture framework described in
npj Comput. Mater. 8, 18 (2022).  We expose class-based APIs so that
other modules (FFT operators, solvers, test harnesses) can depend on a
clean contract when they query energy densities or constraint forces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class PFCParameters:
    """PFC-specific coefficients."""

    r: float = -0.25  # undercooling parameter
    u: float = 0.25  # nonlinear bulk coefficient
    q0: float = 1.0  # principal reciprocal lattice magnitude
    noise: float = 1e-3  # initialization noise amplitude


@dataclass(frozen=True)
class FractureParameters:
    """Phase-field fracture constants inherited from the paper."""

    gc: float = 7.0e2  # J/m^2 initial toughness
    l0: float = 3e-6  # m, regularization length
    k: float = 1e-9  # numerical residual stiffness
    epsilon_half: float = 0.15  # controls toughness decay rate
    gres: float = 0.1  # residual toughness ratio


@dataclass(frozen=True)
class CopperParameters:
    """Cubic elastic constants (GPa) and plasticity inputs for copper."""

    c11: float = 168.4e9
    c12: float = 121.4e9
    c44: float = 75.4e9
    slip_resistance: float = 200e6
    hardening_modulus: float = 10e6
    hardening_b: float = 8.0

    def stiffness_tensor(self, rotation: Array | None = None) -> Array:
        """Return 3x3x3x3 stiffness tensor, rotated if orientation provided."""
        C = np.zeros((3, 3, 3, 3))
        lam = self.c12
        mu = self.c44
        C[0, 0, 0, 0] = C[1, 1, 1, 1] = C[2, 2, 2, 2] = self.c11
        C[0, 0, 1, 1] = C[0, 0, 2, 2] = C[1, 1, 0, 0] = C[1, 1, 2, 2] = C[2, 2, 0, 0] = C[2, 2, 1, 1] = lam
        C[1, 0, 0, 1] = C[0, 1, 1, 0] = C[2, 0, 0, 2] = C[0, 2, 2, 0] = C[2, 1, 1, 2] = C[1, 2, 2, 1] = mu
        C[0, 1, 0, 1] = C[1, 0, 1, 0] = C[0, 2, 0, 2] = C[2, 0, 2, 0] = C[1, 2, 1, 2] = C[2, 1, 2, 1] = mu
        if rotation is None:
            return C
        R = rotation
        C_rot = np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, C, optimize=True)
        return C_rot


class Constraint(Protocol):
    """Protocol describing constraint behavior."""

    def project(self, field: Array) -> Array:
        """Project `field` onto the constraint surface."""


class VolumeConstraint:
    """
    Simple volume conservation constraint for density-like fields.
    """

    def __init__(self, target_mean: float = 0.0) -> None:
        self.target_mean = target_mean

    def project(self, field: Array) -> Array:
        deviation = field.mean() - self.target_mean
        return field - deviation


class PFCCoupling:
    """
    Glue object that maps PFC outputs to the ductile phase-field model.

    mode='density'  -> use the PFC density ψ as a surrogate for damage/plasticity.
    mode='plastic'  -> treat the phase-field plastic strain as the authority and
                       only use ψ to initialize microstructure/defects.
    """

    def __init__(
        self,
        pfc_params: PFCParameters,
        fracture: FractureParameters,
        mode: Literal["density", "plastic"] = "density",
        constraint: Constraint | None = None,
    ) -> None:
        self.pfc_params = pfc_params
        self.fracture = fracture
        self.mode = mode
        self.constraint = constraint or VolumeConstraint()

    def initialize_density(self, shape: Tuple[int, ...], seed: int = 0) -> Array:
        rng = np.random.default_rng(seed)
        psi = self.pfc_params.noise * rng.standard_normal(shape)
        return self.constraint.project(psi)

    def equivalent_plastic_strain(
        self,
        psi: Array,
        plastic_eq: Array | None,
    ) -> Array:
        """
        Translate either ψ or the supplied plastic strain field into ε_eq.
        """
        if self.mode == "plastic" and plastic_eq is not None:
            return plastic_eq
        grad = np.gradient(psi)
        invariant = np.sqrt(sum(g * g for g in grad))
        max_inv = np.max(np.abs(invariant)) + 1e-12
        scaled = np.clip(invariant / max_inv, 0.0, 1.0)
        return np.nan_to_num(scaled)

    def degraded_toughness(self, psi: Array, plastic_eq: Array | None = None) -> Array:
        eps = self.equivalent_plastic_strain(psi, plastic_eq)
        x = eps / self.fracture.epsilon_half
        term = 0.5 - self.fracture.gres * np.tanh(2 * (x - 1.0))
        return self.fracture.gc * (term + 0.5 + self.fracture.gres)


class FreeEnergy:
    """
    Full free-energy functional evaluator with PFC coupling.
    """

    def __init__(
        self,
        copper: CopperParameters,
        fracture: FractureParameters,
        pfc: PFCCoupling,
    ) -> None:
        self.copper = copper
        self.fracture = fracture
        self.pfc = pfc

    def elastic_energy(self, strain: Array, crack: Array, stiffness: Array) -> Array:
        crack_factor = (1.0 - crack) ** 2 + self.fracture.k
        energy_density = 0.5 * np.einsum("...ij,...ijkl,...kl->...", strain, stiffness, strain, optimize=True)
        return crack_factor * energy_density

    def crack_energy(self, crack: Array, toughness: Array) -> Array:
        grad = np.gradient(crack)
        grad_sq = sum(g**2 for g in grad)
        bulk = toughness * (crack**2) / (2 * self.fracture.l0)
        grad_term = toughness * self.fracture.l0 * grad_sq / 2
        return bulk + grad_term

    def positive_strain_energy(self, strain: Array, stiffness: Array) -> Array:
        vals, vecs = np.linalg.eigh(strain)
        vals_pos = np.clip(vals, 0.0, None)
        strain_pos = np.einsum("...ik,...k,...jk->...ij", vecs, vals_pos, vecs, optimize=True)
        return 0.5 * np.einsum("...ij,...ijkl,...kl->...", strain_pos, stiffness, strain_pos, optimize=True)

    def pfc_energy(self, psi: Array) -> Array:
        laplacian = sum(np.gradient(np.gradient(psi, axis=i), axis=i) for i in range(psi.ndim))
        operator = (self.pfc.pfc_params.q0**2 + laplacian) ** 2
        energy = 0.5 * psi * (self.pfc.pfc_params.r + operator) + self.pfc.pfc_params.u * psi**4 / 4
        return np.nan_to_num(energy)

    def total_energy(
        self,
        strain: Array,
        crack: Array,
        psi: Array,
        stiffness: Array,
        plastic_eq: Array | None = None,
    ) -> float:
        toughness = self.pfc.degraded_toughness(psi, plastic_eq)
        elastic = self.elastic_energy(strain, crack, stiffness)
        crack_e = self.crack_energy(crack, toughness)
        pfc_e = self.pfc_energy(psi)
        return float(np.sum(elastic + crack_e + pfc_e))


__all__ = [
    "Array",
    "PFCParameters",
    "FractureParameters",
    "CopperParameters",
    "Constraint",
    "VolumeConstraint",
    "PFCCoupling",
    "FreeEnergy",
]
