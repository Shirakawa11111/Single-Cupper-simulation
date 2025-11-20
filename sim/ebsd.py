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

    def orientation_field(self) -> Array:
        field = np.zeros(self.shape + (3, 3))
        rot = self._rotation_111()
        field[...] = rot
        return field

    def defect_mask(self, seed: int = 0) -> Array:
        rng = np.random.default_rng(seed)
        mask = rng.random(self.shape) < self.defect_fraction
        return mask.astype(float)

    @staticmethod
    def _rotation_111() -> Array:
        v = np.array([1, 1, 1], dtype=float)
        v /= np.linalg.norm(v)
        z = np.array([0, 0, 1], dtype=float)
        axis = np.cross(z, v)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(z @ v)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return rot


__all__ = ["VirtualEBSDGenerator"]
