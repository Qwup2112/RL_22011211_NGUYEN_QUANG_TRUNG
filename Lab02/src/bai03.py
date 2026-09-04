"""Bai 3 - Tinh phan phoi trang thai sau mot buoc.

p_{t+1} = p_t @ P (khong hard-code ket qua).
"""

import numpy as np

from bai01 import P, STATE_NAMES


def next_distribution(p, P) -> np.ndarray:
    """Nhan vector phan phoi voi transition matrix de ra phan phoi ke tiep."""
    return np.asarray(p, dtype=np.float64) @ np.asarray(P, dtype=np.float64)


def main() -> None:
    p0 = np.array([1.0, 0.0, 0.0])   # chac chan bat dau o Sunny

    p1 = next_distribution(p0, P)
    p2 = next_distribution(p1, P)

    print("p0 =", p0, "  (hom nay chac chan Sunny)")
    print("p1 = p0 @ P =", np.round(p1, 6))
    print("p2 = p1 @ P =", np.round(p2, 6))
    print()

    for name, probability in zip(STATE_NAMES, p1):
        print(f"P(ngay mai = {name:<6s}) = {probability:.4f}")

    print()
    print("Tong xac suat p1 =", p1.sum())


if __name__ == "__main__":
    main()
