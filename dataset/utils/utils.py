import numpy as np


def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path}")

        color = header == "PF"

        dim_line = f.readline().decode("ascii").strip()
        while dim_line.startswith("#"):
            dim_line = f.readline().decode("ascii").strip()

        width, height = map(int, dim_line.split())
        scale = float(f.readline().decode("ascii").strip())

        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")

        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data.astype(np.float32)
