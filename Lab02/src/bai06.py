"""Bai 6 - So sanh phan phoi ly thuyet va tan suat mo phong."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bai01 import P, STATE_NAMES  # noqa: E402
from bai04 import state_distribution  # noqa: E402
from bai05 import sample_next_state  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def simulate_frequencies(P, n_transitions: int, start_state: int = 0,
                         seed: int = 42) -> np.ndarray:
    """Mo phong n_transitions buoc va tra ve tan suat tuong doi cua tung state."""
    rng = np.random.default_rng(seed)

    counts = np.zeros(len(STATE_NAMES), dtype=np.int64)
    current_state = start_state

    for _ in range(n_transitions):
        current_state = sample_next_state(current_state, P, rng)
        counts[current_state] += 1

    return counts / counts.sum()


def plot_comparison(theory, empirical, output_path: Path) -> None:
    """Ve bieu do cot so sanh ly thuyet va mo phong."""
    x = np.arange(len(STATE_NAMES))
    width = 0.38

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.bar(x - width / 2, theory, width, label="Ly thuyet (p0 @ P^n)",
             color="#1f77b4")
    axes.bar(x + width / 2, empirical, width, label="Mo phong (tan suat)",
             color="#ff7f0e")

    for index in range(len(STATE_NAMES)):
        axes.text(x[index] - width / 2, theory[index] + 0.005,
                  f"{theory[index]:.4f}", ha="center", va="bottom", fontsize=9)
        axes.text(x[index] + width / 2, empirical[index] + 0.005,
                  f"{empirical[index]:.4f}", ha="center", va="bottom", fontsize=9)

    axes.set_xticks(x, STATE_NAMES)
    axes.set_title("Markov chain: stationary distribution ly thuyet vs mo phong")
    axes.set_xlabel("State")
    axes.set_ylabel("Xac suat")
    axes.grid(True, axis="y", linestyle=":", alpha=0.7)
    axes.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    n_transitions = 100_000

    theory = state_distribution(np.array([1.0, 0.0, 0.0]), P, 200)
    empirical = simulate_frequencies(P, n_transitions)

    print(f"Mo phong {n_transitions} transition, seed = 42")
    print()
    print("State   | Ly thuyet | Mo phong  | Sai lech")
    print("--------+-----------+-----------+---------")
    for index, name in enumerate(STATE_NAMES):
        difference = empirical[index] - theory[index]
        print(f"{name:<8s}| {theory[index]:^10.6f}| {empirical[index]:^10.6f}| "
              f"{difference:+.6f}")

    max_error = float(np.max(np.abs(empirical - theory)))
    print()
    print(f"Sai lech tuyet doi lon nhat: {max_error:.6f}")

    output_path = FIGURES_DIR / "markov_simulation_vs_theory.png"
    plot_comparison(theory, empirical, output_path)
    print(f"Figure saved to: {output_path}")

    # NHAN XET (3-5 dong):
    # 1. Tan suat mo phong khop voi stationary distribution tinh bang phep nhan
    #    ma tran toi 3 chu so thap phan, sai lech chi co mo do ~1e-3.
    # 2. Do la he qua cua luat so lon: chay cang nhieu transition thi tan suat
    #    cang hoi tu ve xac suat ly thuyet (sai so giam theo 1/sqrt(n)).
    # 3. Trang thai xuat phat khong con anh huong sau vai chuc buoc vi chuoi
    #    Markov nay ergodic - moi phan phoi ban dau deu hoi tu ve cung mot ket qua.
    # 4. Day chinh la ly do co the danh gia mot policy bang mo phong (Bai 34)
    #    thay vi phai tinh giai tich.


if __name__ == "__main__":
    main()
