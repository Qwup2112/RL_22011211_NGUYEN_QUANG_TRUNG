"""Bai 1 - Tao transition matrix cho mot Markov chain thoi tiet.

Ba trang thai: Sunny, Cloudy, Rainy.
P[i][j] = xac suat chuyen tu trang thai i sang trang thai j.
"""

import numpy as np

STATE_NAMES = ["Sunny", "Cloudy", "Rainy"]

# Moi hang la phan phoi xac suat tren trang thai ke tiep -> tong moi hang = 1
P = np.array([
    [0.80, 0.15, 0.05],   # tu Sunny
    [0.30, 0.40, 0.30],   # tu Cloudy
    [0.20, 0.30, 0.50],   # tu Rainy
])


def print_transition_matrix(matrix, state_names) -> None:
    """In transition matrix duoi dang bang co ten trang thai."""
    header = "from -> to  | " + " | ".join(f"{name:^7s}" for name in state_names)
    print(header)
    print("-" * len(header))
    for index, name in enumerate(state_names):
        row = " | ".join(f"{value:^7.2f}" for value in matrix[index])
        print(f"{name:<10s} | {row}")


def main() -> None:
    print("Transition matrix P:")
    print(P)
    print()
    print_transition_matrix(P, STATE_NAMES)
    print()
    print("Shape        :", P.shape)
    print("Row sums     :", P.sum(axis=1))
    print("All rows = 1 :", bool(np.allclose(P.sum(axis=1), 1.0)))


if __name__ == "__main__":
    main()
