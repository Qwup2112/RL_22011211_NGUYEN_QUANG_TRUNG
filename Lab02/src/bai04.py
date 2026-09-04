"""Bai 4 - Phan phoi trang thai sau nhieu buoc va bieu do hoi tu."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bai01 import P, STATE_NAMES  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def state_distribution(p0, P, n_steps: int) -> np.ndarray:
    """Tra ve phan phoi trang thai sau n_steps buoc: p0 @ P^n_steps."""
    distribution = np.asarray(p0, dtype=np.float64)
    matrix = np.asarray(P, dtype=np.float64)

    for _ in range(n_steps):
        distribution = distribution @ matrix

    return distribution


def plot_distribution_over_time(p0, P, n_steps: int, output_path: Path) -> None:
    """Ve xac suat cua tung trang thai theo thoi gian."""
    history = np.array([state_distribution(p0, P, step) for step in range(n_steps + 1)])

    figure, axes = plt.subplots(figsize=(10, 5.5))
    colors = ["#ff7f0e", "#7f7f7f", "#1f77b4"]
    for index, name in enumerate(STATE_NAMES):
        axes.plot(range(n_steps + 1), history[:, index], linewidth=2.0,
                  color=colors[index], label=name)
        axes.axhline(history[-1, index], color=colors[index], linestyle=":",
                     linewidth=1.0)

    axes.set_title("Markov chain: phan phoi trang thai hoi tu ve stationary distribution")
    axes.set_xlabel("Buoc thoi gian t")
    axes.set_ylabel("P(state tai buoc t)")
    axes.grid(True, linestyle=":", alpha=0.7)
    axes.legend(title="State")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    p0 = np.array([1.0, 0.0, 0.0])

    print("Phan phoi trang thai theo thoi gian (bat dau tu Sunny):")
    print()
    header = "  t  | " + " | ".join(f"{name:^8s}" for name in STATE_NAMES)
    print(header)
    print("-" * len(header))
    for n_steps in [0, 1, 2, 5, 10, 50]:
        distribution = state_distribution(p0, P, n_steps)
        row = " | ".join(f"{value:^8.6f}" for value in distribution)
        print(f"{n_steps:^5d}| {row}")

    stationary = state_distribution(p0, P, 200)
    print()
    print("Stationary distribution (t = 200):", np.round(stationary, 6))
    print("Khong doi sau mot buoc nua     :",
          np.round(stationary @ P, 6))
    print("Hoi tu:", bool(np.allclose(stationary, stationary @ P, atol=1e-12)))

    # Bat dau tu mot phan phoi khac -> van hoi tu ve cung stationary distribution
    other = state_distribution(np.array([0.0, 0.0, 1.0]), P, 200)
    print("Bat dau tu Rainy, t = 200      :", np.round(other, 6))

    output_path = FIGURES_DIR / "markov_distribution.png"
    plot_distribution_over_time(p0, P, 30, output_path)
    print()
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
