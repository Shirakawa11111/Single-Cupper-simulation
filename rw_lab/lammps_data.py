from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class Atom:
    id: int
    type: int
    position: Tuple[float, float, float]


@dataclass
class AtomType:
    id: int
    mass: float


@dataclass
class SimBox:
    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]


# --- 新增：用于描述键的 dataclass ---
@dataclass
class Bond:
    id: int
    type: int
    atom1_id: int
    atom2_id: int


@dataclass
class BondType:
    id: int


@dataclass
class LammpsData:
    sim_box: SimBox
    atom_types: Dict[int, AtomType]
    atoms: Dict[int, Atom]

    bond_types: Dict[int, BondType] = field(default_factory=dict)
    bonds: Dict[int, Bond] = field(default_factory=dict)

    def write(self, file_name):
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, "w") as f:
            f.write("Position data\n\n")
            f.write(f"{len(self.atoms)} atoms\n")
            f.write(f"{len(self.atom_types)} atom types\n\n")
            f.write(f"{self.sim_box.x[0]} {self.sim_box.x[1]} xlo xhi\n")
            f.write(f"{self.sim_box.y[0]} {self.sim_box.y[1]} ylo yhi\n")
            f.write(f"{self.sim_box.z[0]} {self.sim_box.z[1]} zlo zhi\n\n")
            f.write("Masses\n\n")
            for at in self.atom_types.values():
                f.write(f"{at.id} {at.mass}\n")
            f.write("\nAtoms\n\n")
            for at in self.atoms.values():
                f.write(
                    f"{at.id} {at.type} {at.position[0]} {at.position[1]} {at.position[2]}\n"
                )
