"""
Utilities to create virtual EBSD orientation maps for testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


Array = np.ndarray


@dataclass
class VirtualEBSDGenerator:
    shape: Tuple[int, int, int]
    defect_fraction: float = 0.05
    orientation_vector: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def orientation_field(self) -> Array:
        field = np.zeros(self.shape + (3, 3))
        rot = self._rotation_from_vector(self.orientation_vector)
        field[...] = rot
        return field

    def defect_mask(self, seed: int = 0) -> Array:
        rng = np.random.default_rng(seed)
        mask = rng.random(self.shape) < self.defect_fraction
        return mask.astype(float)

    @staticmethod
    def _rotation_from_vector(direction: Tuple[float, float, float]) -> Array:
        v = np.array(direction, dtype=float)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            raise ValueError("orientation_vector must be non-zero")
        v /= norm
        z = np.array([0.0, 0.0, 1.0], dtype=float)
        if np.allclose(v, z):
            return np.eye(3)
        if np.allclose(v, -z):
            # 180-degree rotation about x-axis
            return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        axis = np.cross(z, v)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            return np.eye(3)
        axis /= axis_norm
        angle = np.arccos(np.clip(z @ v, -1.0, 1.0))
        K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return rot


__all__ = ["VirtualEBSDGenerator"]
