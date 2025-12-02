"""
Energy models for coupling a phase-field fracture solver with a
Phase-Field Crystal (PFC) density field.

The design follows the ductile-fracture framework described in
npj Comput. Mater. 8, 18 (2022).  We expose class-based APIs so that
other modules (FFT operators, solvers, test harnesses) can depend on a
clean contract when they query energy densities or constraint forces.

Current defaults are **fully non-dimensional**: length is scaled by the
grid spacing (Δx = 1), stress by the reference copper stiffness c11, and
energy by (stress × length³). If you need dimensional runs, rescale the
material and fracture parameters accordingly before constructing the solver.
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
    """Phase-field fracture constants (non-dimensional)."""

    gc: float = 1.0  # toughness scale, nondimensional
    l0: float = 1.0  # regularization length scaled by grid spacing
    k: float = 1.0e-6  # numerical residual stiffness
    epsilon_half: float = 0.15  # controls toughness decay rate
    gres: float = 0.1  # residual toughness ratio


@dataclass(frozen=True)
class CopperParameters:
    """
    Cubic elastic constants and plasticity inputs for copper
    in **non-dimensional form**. Stiffness values are normalized by
    the reference c11 (physical c11 = 168.4 GPa), so all stresses
    are σ* = σ_phys / 168.4 GPa. Physical values: c11=168.4 GPa,
    c12=121.4 GPa, c44=75.4 GPa (E_[111]≈191 GPa, ~3×E_[100]).
    Plastic parameters are scaled by the same stress unit.
    """

    c11: float = 1.0
    c12: float = 0.7209  # 121.4 / 168.4
    c44: float = 0.4477  # 75.4 / 168.4
    slip_resistance: float = 1.07e-3  # 180 MPa / 168.4 GPa
    hardening_modulus: float = 6.0e-5  # ~10 MPa / 168.4 GPa
    hardening_b: float = 8.0
    residual_stiffness: float = 1e-6  # crack residual stiffness

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
        # yield stress scaled by c11_ref=168.4 GPa; 180 MPa -> ~1.07e-3
        yield_tau: float = 1.07e-3,
        # flow scaling (nd): smaller slows post-yield softening; tune vs σ–ε
        flow_scale: float = 5e-4,
        visco_exponent: float = 10.0,  # nonlinear viscoplasticity (stable overstress)
        visco_ref: float | None = None,
        # isotropic hardening slope (nd): ~10 MPa / 168.4 GPa ≈ 6e-5
        linear_hardening: float = 6.0e-5,
    ) -> None:
        self.pfc_params = pfc_params
        self.fracture = fracture
        self.mode = mode
        self.constraint = constraint or VolumeConstraint()
        self.yield_tau = yield_tau
        self.flow_scale = flow_scale
        self.visco_exponent = visco_exponent
        self.visco_ref = visco_ref
        self.linear_hardening = linear_hardening
        # Precompute valid FCC slip systems (n, m) with m·n=0
        normals = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        )
        dirs = np.array(
            [
                [0, 1, -1],
                [0, 1, 1],
                [1, 0, -1],
                [1, 0, 1],
                [1, -1, 0],
                [1, 1, 0],
            ],
            dtype=float,
        )
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        self.slip_systems = []
        for n in normals:
            for m in dirs:
                if np.abs(np.dot(m, n)) < 1e-6:  # ensure orthogonality
                    self.slip_systems.append((m, n))

    def initialize_density(self, shape: Tuple[int, ...], seed: int = 0) -> Array:
        rng = np.random.default_rng(seed)
        psi = self.pfc_params.noise * rng.standard_normal(shape)
        return self.constraint.project(psi)

    def plastic_measures(
        self,
        psi: Array,
        plastic_eq: Array | None,
        plastic_vec: Array | None = None,
        strain: Array | None = None,
        stress: Array | None = None,
        backstress: Array | None = None,
        load_axis: int = 0,
        mech_weight: float = 0.7,
    ) -> tuple[Array, Array, Array]:
        """
        Compute scalar equivalent plastic strain, directional surrogate, and a
        plastic strain tensor proxy using RSS on FCC slip systems.

        Returns (eps_eq_scalar, eps_vec, epsp_tensor) where:
          - eps_eq_scalar: equivalent plastic proxy
          - eps_vec: directional proxy (|.|<=1)
          - epsp_tensor: symmetric plastic strain proxy tensor matching strain shape
        """
        mech_weight = np.clip(mech_weight, 0.0, 1.0)
        # If user supplies authoritative plastic data, return it
        if self.mode == "plastic" and plastic_eq is not None and plastic_vec is not None:
            epsp_tensor = np.zeros(plastic_vec.shape[:-1] + (3, 3))
            return plastic_eq, plastic_vec, epsp_tensor

        # --- PFC-based measures ---
        grad = np.gradient(psi)
        inv = np.sqrt(sum(g * g for g in grad))
        max_inv = np.max(np.abs(inv)) + 1e-12
        eps_eq_pfc = np.clip(inv / max_inv, 0.0, 1.0)
        max_comp = max(np.max(np.abs(g)) for g in grad) + 1e-12
        comp_pfc = [np.clip(np.abs(g) / max_comp, 0.0, 1.0) for g in grad]

        # --- Mechanical-based measures using RSS on FCC slip systems ---
        eps_eq_mech = np.zeros_like(psi)
        comp_mech = [np.zeros_like(psi) for _ in range(3)]
        epsp_tensor = np.zeros(psi.shape + (3, 3))

        if strain is not None:
            # resolved shear stress proxy: use strain as proxy (small-strain)
            # Prefer stress if provided; else use strain. For each slip system,
            # compute RSS = m · ((stress-backstress) · n) or m · (strain · n).
            rss_list = []
            mn_list = []
            proj_signs = []
            tensor = None
            if stress is not None:
                tensor = stress if backstress is None else stress - backstress
            elif strain is not None:
                tensor = strain
            for m, n in self.slip_systems:
                mn = np.outer(m, n)
                mn_list.append(mn)
                proj = np.einsum("...ij,ij->...", tensor, mn, optimize=True)
                rss_list.append(np.abs(proj))
                proj_signs.append(np.sign(proj))
            rss_stack = np.stack(rss_list, axis=0)
            rss_max = np.max(rss_stack, axis=0)
            # Stable over-stress + linear isotropic hardening:
            # raise yield with accumulated plastic_eq, and bound flow using a saturating overstress ratio
            yield_eff = self.yield_tau
            if plastic_eq is not None:
                yield_eff = yield_eff + self.linear_hardening * np.clip(plastic_eq, 0.0, 1.0)
            yield_eff = np.clip(yield_eff, self.yield_tau, None)
            ref = self.visco_ref if self.visco_ref is not None else yield_eff
            overstress = np.clip(rss_max - yield_eff, 0.0, None)
            ratio = overstress / (ref + overstress + 1e-12)
            flow_rate = (ratio) ** self.visco_exponent
            flow_scaled = flow_rate * self.flow_scale
            eps_eq_mech = np.clip(flow_scaled, 0.0, 1.0)
            # Directional: use the slip system achieving max RSS
            idx_max = np.argmax(rss_stack, axis=0)
            comp_mech = [np.zeros_like(psi) for _ in range(3)]
            # Build directional vector from winning slip direction components
            for k, (m, n) in enumerate(self.slip_systems):
                selector = (idx_max == k)
                if np.any(selector):
                    comp_mech[0][selector] = np.abs(m[0])
                    comp_mech[1][selector] = np.abs(m[1])
                    comp_mech[2][selector] = np.abs(m[2])
            # Build a plastic strain proxy tensor from the winning slip system mn
            # For simplicity, assign epsp_tensor = flow * sign(proj) * sym(m⊗n)
            epsp_tensor = np.zeros_like(strain)
            for k in range(len(mn_list)):
                selector = (idx_max == k)
                if np.any(selector):
                    mn = mn_list[k]
                    sym_mn = 0.5 * (mn + mn.T)
                    sign_k = proj_signs[k]
                    epsp_tensor[selector] = flow_scaled[selector][..., None, None] * sym_mn * sign_k[selector][..., None, None]

        # Blend mechanical and PFC contributions for scalar/directional proxies
        eps_eq = mech_weight * eps_eq_mech + (1.0 - mech_weight) * eps_eq_pfc
        comp_blend = [
            mech_weight * cm + (1.0 - mech_weight) * cp for cm, cp in zip(comp_mech, comp_pfc)
        ]
        eps_vec = np.stack(comp_blend, axis=-1)

        eps_eq = np.clip(np.nan_to_num(eps_eq), 0.0, 1.0)
        eps_vec = np.nan_to_num(eps_vec)
        epsp_tensor = np.nan_to_num(epsp_tensor)
        return eps_eq, eps_vec, epsp_tensor

    def degraded_toughness(
        self,
        psi: Array,
        plastic_eq: Array | None = None,
        grain_mask: Array | None = None,
        grain_scale: float = 0.5,
    ) -> Array:
        # plastic_eq 应传入累积等效塑性；若为空则回退到 PFC proxy
        if plastic_eq is None:
            plastic_eq, _, _ = self.plastic_measures(psi, None, None)
        x = plastic_eq / self.fracture.epsilon_half
        term = 0.5 - self.fracture.gres * np.tanh(2 * (x - 1.0))
        gc_eff = self.fracture.gc * (term + 0.5 + self.fracture.gres)
        if grain_mask is not None:
            gc_eff = gc_eff * (1.0 - grain_scale * grain_mask)
        return gc_eff


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
        grain_mask: Array | None = None,
        plastic_tensor: Array | None = None,
    ) -> float:
        toughness = self.pfc.degraded_toughness(psi, plastic_eq, grain_mask=grain_mask)
        strain_eff = strain if plastic_tensor is None else strain - plastic_tensor
        elastic = self.elastic_energy(strain_eff, crack, stiffness)
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
