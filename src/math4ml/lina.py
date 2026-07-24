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


def rref(a: NDArray) -> NDArray:
    a = a.copy()
    n_rows = a.shape[0]
    n_cols = a.shape[1]

    i = 0
    for j in range(min(n_cols, n_rows)):
        r = find_first_nonzero(a[i:, j])
        if r is not None:
            a = swap_rows(a, i, i + r)
            a = scale_row(a, i, j)
            a = subtract_row(a, i, j)
            i += 1

    return a


def is_ref(a: NDArray) -> bool:
    last_pivot = -1
    for i in range(a.shape[0]):
        pivot = find_first_nonzero(a[i])
        # pivots must be to the right of all previous pivots
        if pivot is not None:
            if pivot <= last_pivot:
                return False
            last_pivot = pivot
        else:
            last_pivot = a.shape[1]
    return True


def is_rref(a: NDArray) -> bool:
    # 1. matrix must be in REF
    if not is_ref(a):
        return False
    for i in range(a.shape[0]):
        pivot = find_first_nonzero(a[i])
        if pivot is not None:
            # 2. pivots must be 1
            if a[i, pivot] != 1:
                return False
            # 3. pivot column only contains 0
            for j in range(0, pivot):
                if a[i, j] != 0:
                    return False
    return True


def minus_one_trick(x: NDArray, y: NDArray = None) -> tuple[NDArray, NDArray]:
    if y is None:
        y = np.zeros(x.shape[0])
    x_rows = []
    y_rows = []

    i = 0
    for j in range(min(x.shape[0], x.shape[1])):
        if x[i, j] == 0:
            row = np.zeros(x.shape[1])
            row[j] = -1
            x_rows.append(row)
            y_rows.append([0])
        else:
            x_rows.append(x[i])
            y_rows.append([y[i]])
            i += 1
    if i < x.shape[0]:
        x_rows.append(x[i:])
        y_rows.append(y[i:])
    xm1 = np.vstack(x_rows)
    ym1 = np.concat(y_rows)

    return xm1, ym1


def solve(x: NDArray, y: NDArray = None):
    if y is None:
        y = np.zeros(x.shape[0])

    a = np.column_stack([x, y])
    a = rref(a)

    x = a[:, :-1]
    y = a[:, -1]
    x, y = minus_one_trick(x, y)

    solution = []
    space = []
    for j in range(x.shape[1]):
        if x[j, j] == 1:
            solution.append(y[j])
        else:
            solution.append(0)
            space.append(x[:x.shape[1], j])
    solution = np.array(solution)

    return x, y, solution, space
