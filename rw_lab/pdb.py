from dataclasses import dataclass
from typing import Self
from pathlib import Path


@dataclass
class Pdb:
    """pdb形式の読み書き

    現在は炭素原子のみ、書き込みのみ可能

    """

    title: str
    atoms: dict[int, tuple[float, float, float]]

    def write(self, file_name: str):
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, mode="w") as f:
            f.write("HEADER\n")
            f.write(f"TITLE  {self.title}\n")
            for k, v in self.atoms.items():
                f.write(
                    f"HETATM {k:>{4}}  C   UNK  0001    {v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}\n"
                )
        return
