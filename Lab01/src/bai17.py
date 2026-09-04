"""Bai 17 - Ve reward theo episode va luu hinh vao Lab01/figures/."""

from pathlib import Path

import matplotlib

# Dung backend "Agg" de script chay duoc ca khi khong co man hinh do hoa
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bai14 import run_random_episodes  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def plot_episode_rewards(episode_rewards: list[float], output_path: Path) -> None:
    """Ve do thi reward theo episode (co title, xlabel, ylabel, grid)."""
    episodes = np.arange(1, len(episode_rewards) + 1)

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(episodes, episode_rewards, marker="o", markersize=3,
              linewidth=1.2, color="#1f77b4", label="Total reward")

    mean_reward = float(np.mean(episode_rewards))
    axes.axhline(mean_reward, color="#d62728", linestyle="--", linewidth=1.2,
                 label=f"Mean = {mean_reward:.2f}")

    axes.set_title("Random agent on CartPole-v1: total reward per episode")
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

    output_path = FIGURES_DIR / "reward_cartpole.png"
    plot_episode_rewards(episode_rewards, output_path)

    print(f"Episodes plotted : {len(episode_rewards)}")
    print(f"Mean reward      : {np.mean(episode_rewards):.2f}")
    print(f"Figure saved to  : {output_path}")


if __name__ == "__main__":
    main()
