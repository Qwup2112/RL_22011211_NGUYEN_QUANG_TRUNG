"""Bai 11 - Reward som va reward tre: khi nao B tot hon A?

sequence_A = [5, 0, 0, 0, 0]  -> G_A = 5           (khong phu thuoc gamma)
sequence_B = [0, 0, 0, 0, 10] -> G_B = 10*gamma^4  (phu thuoc manh vao gamma)
"""

import numpy as np

from bai07 import compute_return

sequence_A = [5, 0, 0, 0, 0]
sequence_B = [0, 0, 0, 0, 10]


def find_crossover_gamma(sequence_A, sequence_B,
                         tolerance: float = 1e-12) -> float:
    """Tim gamma tai do return cua B bang return cua A (chia doi khoang).

    Ham f(gamma) = G_B(gamma) - G_A(gamma) tang don dieu tren [0, 1] voi hai
    chuoi nay, nen chia doi khoang la du.
    """
    def difference(gamma: float) -> float:
        return compute_return(sequence_B, gamma) - compute_return(sequence_A, gamma)

    low, high = 0.0, 1.0
    if difference(high) <= 0:
        return float("nan")      # B khong bao gio vuot A

    while high - low > tolerance:
        middle = (low + high) / 2
        if difference(middle) > 0:
            high = middle
        else:
            low = middle

    return (low + high) / 2


def main() -> None:
    print("sequence_A =", sequence_A)
    print("sequence_B =", sequence_B)
    print()
    print("| Gamma | G_A    | G_B    | B > A |")
    print("|------:|-------:|-------:|:-----:|")
    for gamma in [0.0, 0.5, 0.8, 0.84, 0.85, 0.9, 0.99, 1.0]:
        G_A = compute_return(sequence_A, gamma)
        G_B = compute_return(sequence_B, gamma)
        print(f"| {gamma:5.2f} | {G_A:6.3f} | {G_B:6.3f} | "
              f"{str(G_B > G_A):^5s} |")

    # Quet mot luoi min de tim khoang gamma ma B thang
    gammas = np.linspace(0, 1, 100001)
    better = np.array([compute_return(sequence_B, g) > compute_return(sequence_A, g)
                       for g in gammas])
    first_better = float(gammas[np.argmax(better)]) if better.any() else float("nan")

    crossover = find_crossover_gamma(sequence_A, sequence_B)

    print()
    print(f"Quet luoi 100001 diem : B tot hon A tu gamma ~ {first_better:.5f}")
    print(f"Chia doi khoang       : diem hoa von gamma* = {crossover:.10f}")
    print(f"Nghiem giai tich      : gamma* = 0.5^(1/4)  = {0.5 ** 0.25:.10f}")
    print()
    print(f"=> B co return lon hon A khi gamma > {crossover:.4f} "
          f"(va bang nhau tai gamma*).")
    print()
    print("Y nghia: gamma quyet dinh agent 'kien nhan' den dau. Agent thien can")
    print("(gamma nho) chon phan thuong 5 ngay lap tuc; agent kien nhan (gamma lon)")
    print("chap nhan doi 4 buoc de lay phan thuong 10.")


if __name__ == "__main__":
    main()
