import numpy as np
from numpy.typing import NDArray


def array_to_math(a: NDArray, sep="\t") -> str:
    text = "$$\n\\begin{bmatrix}\n"
    for row in a:
        row = row.tolist()
        for col in range(len(row)):
            text += f"{sep}"
            if col != 0:
                text += f"& "
            item = row[col]
            text += f"{item:.3g}"
        text += f"{sep}\\\\\n"
    text += "\\end{bmatrix}\n$$"
    return text
