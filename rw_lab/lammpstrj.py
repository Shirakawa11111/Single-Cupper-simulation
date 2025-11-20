from dataclasses import dataclass
from typing import Self
from pathlib import Path


@dataclass
class Lammpstrj:
    timestep: int
    # ((xlo, xhi), (ylo, yhi), (zlo, zhi))
    simbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    atoms: dict[int, dict[str, float]]

    @classmethod
    def read(cls, file_name: str | Path) -> Self:
        timestep = None
        simbox = None
        atoms = {}
        with open(file_name) as f:
            while True:
                line = f.readline()
                match line:
                    case l if l.startswith("ITEM: TIMESTEP"):
                        timestep = int(f.readline().strip())
                    case l if l.startswith("ITEM: BOX BOUNDS"):
                        x_box = f.readline().strip().split()
                        x_lo = float(x_box[0])
                        x_hi = float(x_box[1])
                        y_box = f.readline().strip().split()
                        y_lo = float(y_box[0])
                        y_hi = float(y_box[1])
                        z_box = f.readline().strip().split()
                        z_lo = float(z_box[0])
                        z_hi = float(z_box[1])
                        simbox = ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))
                    case l if l.startswith("ITEM: NUMBER OF ATOMS"):
                        f.readline()
                    case l if l.startswith("ITEM: ATOMS"):
                        col_names = l.strip().split()[2:]
                        for line in f:
                            values = line.strip().split()
                            id_key = int(float(values[col_names.index("id")]))
                            atoms[id_key] = {
                                col_names[i]: float(values[i])
                                for i in range(len(values))
                            }

                    case l if not l:
                        # EOF
                        break
                    case l:
                        raise Exception(
                            "unintended line", "file broken or unimplemented", l
                        )

        if timestep is None:
            raise Exception("file does not contain timestep")
        if simbox is None:
            raise Exception("file does not contain simulation box")
        return Lammpstrj(timestep, simbox, atoms)

    def write(self, file_name: str):
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, mode="w") as f:
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{self.timestep}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(self.atoms)}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"{self.simbox[0][0]} {self.simbox[0][1]}\n")
            f.write(f"{self.simbox[1][0]} {self.simbox[1][1]}\n")
            f.write(f"{self.simbox[2][0]} {self.simbox[2][1]}\n")
            labels = list(next(iter(self.atoms.values())).keys())
            f.write(f"ITEM: ATOMS {" ".join(labels)}\n")
            for _, row in self.atoms.items():
                values = []
                for label in labels:
                    values.append(row[label])
                f.write(f"{" ".join(map(str,values))}\n")
        return
