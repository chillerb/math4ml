import numpy as np

from numpy.typing import NDArray


def find_first_nonzero(a: NDArray) -> int:
    for i in range(len(a)):
        if a[i] != 0:
            return i
    return None


def swap_rows(a: NDArray, row_1: int, row_2: int) -> NDArray:
    row = a[row_1]
    a[row_1] = a[row_2]
    a[row_2] = row
    return a


def scale_row(a: NDArray, row: int, column: int):
    a[row] = a[row] / a[row, column]
    return a


def subtract_row(a: NDArray, row: int, column: int) -> NDArray:
    n_rows = a.shape[0]
    for i in range(n_rows):
        if i != row:
            a[i] = a[i] - a[i, column] * a[row]
    return a


def rref(a: NDArray):
    a = a.copy()
    n_rows = a.shape[0]
    n_cols = a.shape[1]

    for j in range(min(n_cols, n_rows)):
        r = find_first_nonzero(a[j:, j])
        if r is not None:
            a = swap_rows(a, j, j + r)
            a = scale_row(a, j, j)
            a = subtract_row(a, j, j)

    return a
