"""Bai 18 - Moving average cua reward (tu cai dat, khong dung Pandas rolling)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bai14 import run_random_episodes  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def moving_average(values, window_size: int) -> list[float]:
    """Tinh trung binh truot voi cua so window_size.

    Tra ve list co do dai len(values) - window_size + 1.
    Cai dat thu cong bang vong lap, khong dung pandas.rolling.
    """
    if window_size <= 0:
        raise ValueError("window_size phai la so nguyen duong")
    if window_size > len(values):
        raise ValueError("window_size khong duoc lon hon so phan tu cua values")

    averages = []
    running_sum = float(sum(values[:window_size]))
    averages.append(running_sum / window_size)

    # Truot cua so: cong phan tu moi, tru phan tu vua roi ra khoi cua so
    for index in range(window_size, len(values)):
        running_sum += float(values[index]) - float(values[index - window_size])
        averages.append(running_sum / window_size)

    return averages


def plot_reward_and_moving_average(episode_rewards: list[float],
                                   window_size: int,
                                   output_path: Path) -> None:
    """Ve dong thoi reward goc va duong moving average."""
    episodes = np.arange(1, len(episode_rewards) + 1)
    smoothed = moving_average(episode_rewards, window_size)

    # Diem moving average dau tien ung voi episode thu window_size
    smoothed_x = np.arange(window_size, len(episode_rewards) + 1)

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(episodes, episode_rewards, color="#aec7e8", linewidth=1.0,
              marker="o", markersize=3, label="Raw reward")
    axes.plot(smoothed_x, smoothed, color="#d62728", linewidth=2.0,
              label=f"Moving average (window = {window_size})")

    axes.set_title("CartPole-v1 random agent: reward and moving average")
    axes.set_xlabel("Episode")
    axes.set_ylabel("Total reward")
    axes.grid(True, linestyle=":", alpha=0.7)
    axes.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    episode_rewards, _ = run_random_episodes(n_episodes=100, seed=42)

    window_size = 10
    smoothed = moving_average(episode_rewards, window_size)

    print(f"Number of raw values      : {len(episode_rewards)}")
    print(f"Window size               : {window_size}")
    print(f"Number of averaged values : {len(smoothed)}")
    print()
    print("First 5 moving averages:", [f"{value:.2f}" for value in smoothed[:5]])
    print("Last 5 moving averages :", [f"{value:.2f}" for value in smoothed[-5:]])

    output_path = FIGURES_DIR / "moving_average.png"
    plot_reward_and_moving_average(episode_rewards, window_size, output_path)
    print()
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
