"""
Builders for synthetic single-crystal copper configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .ebsd import VirtualEBSDGenerator
from .io import write_atomic_data, write_lammpstrj
from .operators import GridSpec


@dataclass
class Cu111Structure:
    grid: GridSpec
    fields: Dict[str, np.ndarray]

    def export(self, data_path, dump_path, timestep: int = 0) -> None:
        write_atomic_data(data_path, self.grid)
        write_lammpstrj(dump_path, self.grid, self.fields, timestep)


class Cu111StructureBuilder:
    """
    Creates a synthetic 111-oriented single crystal with random defects.
    """

    def __init__(
        self,
        grid: GridSpec,
        defect_fraction: float = 0.05,
        defect_amplitude: float = 0.2,
        noise: float = 1e-3,
    ) -> None:
        self.grid = grid
        self.defect_fraction = defect_fraction
        self.defect_amplitude = defect_amplitude
        self.noise = noise

    def build(self, seed: int = 0) -> Cu111Structure:
        generator = VirtualEBSDGenerator(self.grid.shape, self.defect_fraction)
        mask = generator.defect_mask(seed)
        rng = np.random.default_rng(seed)
        psi = self.noise * rng.standard_normal(self.grid.shape) + self.defect_amplitude * mask
        crack = np.zeros(self.grid.shape)
        plastic = mask * 0.0
        fields = {"psi": psi, "crack": crack, "plastic": plastic}
        return Cu111Structure(self.grid, fields)
