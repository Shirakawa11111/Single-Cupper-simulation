"""
Builders for synthetic single-crystal copper configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from scipy.ndimage import gaussian_gradient_magnitude
from .ebsd import VirtualEBSDGenerator
from .defects import DefectConfig, generate_defect_seeds, seeds_to_fields
from .io import write_atomic_data, write_lammpstrj
from .operators import GridSpec


@dataclass
class Cu111Structure:
    grid: GridSpec
    fields: Dict[str, np.ndarray]
    orientation: np.ndarray
    grain_mask: np.ndarray | None = None

    def export(self, data_path, dump_path, timestep: int = 0) -> None:
        write_atomic_data(data_path, self.grid)
        write_lammpstrj(dump_path, self.grid, self.fields, timestep)


class Cu111StructureBuilder:
    """
    Creates synthetic copper structures with optional multigrain support.

    Use cases:
    - Default: single-crystal [111] with random defects.
    - Multigrain: provide grain_labels + grain_orientations (dict or array)
      to assign per-grain rotations and inject boundary defects.
    - Direct orientation map: supply orientation_map with shape (*grid.shape,3,3).
    """

    def __init__(
        self,
        grid: GridSpec,
        defect_fraction: float = 0.05,
        defect_amplitude: float = 0.2,
        noise: float = 1e-3,
        orientation_vector: tuple[float, float, float] | None = None,
        orientation_map: np.ndarray | None = None,
        grain_labels: np.ndarray | None = None,
        grain_orientations: Dict[int, np.ndarray] | np.ndarray | None = None,
        boundary_amplitude: float = 0.3,
        boundary_width: int = 1,
        boundary_misorientation_deg: float = 5.0,
        defect_config: Dict | None = None,
    ) -> None:
        self.grid = grid
        self.defect_fraction = defect_fraction
        self.defect_amplitude = defect_amplitude
        self.noise = noise
        self.orientation_vector = orientation_vector
        self.orientation_map = orientation_map
        self.grain_labels = grain_labels
        self.grain_orientations = grain_orientations
        self.boundary_amplitude = boundary_amplitude
        self.boundary_width = max(1, boundary_width)
        self.boundary_misorientation = np.deg2rad(boundary_misorientation_deg)
        self.defect_config = defect_config

    def _build_orientation(self) -> np.ndarray:
        """Construct orientation field from provided inputs or default [111]."""
        shape = self.grid.shape + (3, 3)
        # 1) Direct orientation map
        if self.orientation_map is not None:
            if self.orientation_map.shape != shape:
                raise ValueError(f"orientation_map shape {self.orientation_map.shape} != {shape}")
            return np.array(self.orientation_map, copy=True)
        # 2) Grain labels + per-grain orientations
        if self.grain_labels is not None and self.grain_orientations is not None:
            labels = np.asarray(self.grain_labels)
            if labels.shape != self.grid.shape:
                raise ValueError(f"grain_labels shape {labels.shape} != grid shape {self.grid.shape}")
            orientations = np.zeros(shape)
            unique_labels = np.unique(labels)
            def _fetch_orientation(label: int) -> np.ndarray:
                if isinstance(self.grain_orientations, dict):
                    if label not in self.grain_orientations:
                        raise ValueError(f"grain_orientations missing label {label}")
                    return np.asarray(self.grain_orientations[label])
                arr = np.asarray(self.grain_orientations)
                if arr.ndim != 3 or arr.shape[1:] != (3, 3):
                    raise ValueError("grain_orientations array must be (n_grains,3,3)")
                if label >= arr.shape[0]:
                    raise ValueError(f"grain_orientations array length {arr.shape[0]} missing label {label}")
                return arr[label]
            for lbl in unique_labels:
                orientations[labels == lbl] = _fetch_orientation(int(lbl))
            return orientations
        # 3) Fallback: single [111] (or user-specified)
        generator = VirtualEBSDGenerator(
            self.grid.shape,
            self.defect_fraction,
            orientation_vector=self.orientation_vector or (1.0, 1.0, 1.0),
        )
        return generator.orientation_field()

    def _grain_boundary_mask(self, orientations: np.ndarray) -> np.ndarray:
        """
        Detect grain boundaries either from labels (if provided) or
        misorientation angle between neighbors exceeding threshold.
        """
        if self.grain_labels is not None:
            labels = np.asarray(self.grain_labels)
            boundary = np.zeros_like(labels, dtype=bool)
            for axis, periodic in enumerate(self.grid.periodic):
                rolled = np.roll(labels, -1, axis=axis)
                if not periodic:
                    # avoid wrap-around artifacts on the last slice
                    boundary_slice = [slice(None)] * labels.ndim
                    boundary_slice[axis] = -1
                    rolled[tuple(boundary_slice)] = labels[tuple(boundary_slice)]
                boundary |= labels != rolled
            # optional dilation to widen boundary region
            for _ in range(self.boundary_width - 1):
                expanded = boundary.copy()
                for axis, periodic in enumerate(self.grid.periodic):
                    expanded |= np.roll(boundary, 1, axis=axis)
                    expanded |= np.roll(boundary, -1, axis=axis)
                    if not periodic:
                        head = [slice(None)] * labels.ndim
                        head[axis] = 0
                        tail = [slice(None)] * labels.ndim
                        tail[axis] = -1
                        expanded[tuple(head)] = boundary[tuple(head)]
                        expanded[tuple(tail)] = boundary[tuple(tail)]
                boundary = expanded
            return boundary.astype(float)

        # Misorientation-based detection
        boundary = np.zeros(self.grid.shape, dtype=bool)
        threshold = self.boundary_misorientation
        for axis, periodic in enumerate(self.grid.periodic):
            neighbor = np.roll(orientations, -1, axis=axis)
            if not periodic:
                # avoid wrap boundary artifacts
                slicer = [slice(None)] * orientations.ndim
                slicer[axis] = -1
                neighbor[tuple(slicer)] = orientations[tuple(slicer)]
            rel = np.matmul(orientations, np.swapaxes(neighbor, -1, -2))
            cos_angle = (np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            boundary |= angle > threshold
        for _ in range(self.boundary_width - 1):
            expanded = boundary.copy()
            for axis, periodic in enumerate(self.grid.periodic):
                expanded |= np.roll(boundary, 1, axis=axis)
                expanded |= np.roll(boundary, -1, axis=axis)
                if not periodic:
                    head = [slice(None)] * boundary.ndim
                    head[axis] = 0
                    tail = [slice(None)] * boundary.ndim
                    tail[axis] = -1
                    expanded[tuple(head)] = boundary[tuple(head)]
                    expanded[tuple(tail)] = boundary[tuple(tail)]
            boundary = expanded
        return boundary.astype(float)

    def _grain_boundary_mask_from_orientation(self, orientation: np.ndarray, sigma: float = 1.0, threshold: float = 0.1) -> np.ndarray:
        """
        Approximate grain boundary mask from orientation gradients.
        Uses a Gaussian-smoothed gradient magnitude of orientation vectors.
        """
        # use orientation as 9-component field
        comps = [orientation[..., i, j] for i in range(3) for j in range(3)]
        grad_mag_sum = np.zeros(self.grid.shape)
        for comp in comps:
            gm = gaussian_gradient_magnitude(comp, sigma=sigma)
            grad_mag_sum += gm
        norm = grad_mag_sum / (np.max(grad_mag_sum) + 1e-12)
        mask = (norm > threshold).astype(float)
        return mask

    def build(self, seed: int = 0) -> Cu111Structure:
        orientation = self._build_orientation()
        rng = np.random.default_rng(seed)
        grain_mask = self._grain_boundary_mask_from_orientation(orientation, sigma=1.0, threshold=0.1)

        use_seeded_defects = self.defect_config is not None and len(self.defect_config) > 0
        if use_seeded_defects:
            cfg = DefectConfig(**self.defect_config)
            seeds = generate_defect_seeds(self.grid, cfg, rng)
            seeded = seeds_to_fields(self.grid, seeds, cfg)
            # add a small noise floor for numerical stability
            psi = seeded["psi"] + self.noise * rng.standard_normal(self.grid.shape)
            crack = seeded["crack"]
            plastic = seeded["plastic"]
        else:
            mask = np.zeros(self.grid.shape)
            if self.defect_fraction > 0:
                generator = VirtualEBSDGenerator(self.grid.shape, self.defect_fraction)
                mask = generator.defect_mask(seed)

            boundary_mask = np.zeros(self.grid.shape)
            if self.boundary_amplitude > 0:
                boundary_mask = self._grain_boundary_mask(orientation)

            psi = (
                self.noise * rng.standard_normal(self.grid.shape)
                + self.defect_amplitude * mask
                + self.boundary_amplitude * boundary_mask
            )
            crack = np.zeros(self.grid.shape)
            plastic = mask * 0.0

        fields = {"psi": psi, "crack": crack, "plastic": plastic}
        return Cu111Structure(self.grid, fields, orientation, grain_mask=grain_mask)
