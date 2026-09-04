"""Bai 2 - Kiem tra tinh hop le cua mot transition matrix."""

import numpy as np

from bai01 import P


def validate_transition_matrix(P, tol: float = 1e-10) -> bool:
    """Kiem tra P co phai transition matrix hop le hay khong.

    1. P la ma tran vuong;
    2. moi phan tu thuoc [0, 1];
    3. tong moi hang xap xi 1 (sai so <= tol).
    """
    matrix = np.asarray(P, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False

    if np.any(matrix < -tol) or np.any(matrix > 1.0 + tol):
        return False

    row_sums = matrix.sum(axis=1)
    if not np.all(np.abs(row_sums - 1.0) <= max(tol, 1e-12)):
        return False

    return True


def explain(name: str, matrix) -> None:
    """In ket qua kiem tra kem ly do."""
    matrix = np.asarray(matrix, dtype=np.float64)
    valid = validate_transition_matrix(matrix)
    print(f"{name:<28s}: {valid}")

    if not valid:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            print(f"    -> khong vuong, shape = {matrix.shape}")
        elif np.any(matrix < 0) or np.any(matrix > 1):
            print("    -> co phan tu nam ngoai [0, 1]")
        else:
            print(f"    -> tong hang = {matrix.sum(axis=1)}")


def main() -> None:
    print("Kiem tra transition matrix:")
    print()

    explain("P hop le (Bai 1)", P)

    # Cac truong hop sai de kiem thu
    explain("khong vuong", np.array([[0.5, 0.5, 0.0], [0.2, 0.8, 0.0]]))
    explain("co phan tu am", np.array([[1.2, -0.2], [0.5, 0.5]]))
    explain("tong hang khac 1", np.array([[0.5, 0.4], [0.5, 0.5]]))
    explain("ma tran don vi 3x3", np.eye(3))


if __name__ == "__main__":
    main()
