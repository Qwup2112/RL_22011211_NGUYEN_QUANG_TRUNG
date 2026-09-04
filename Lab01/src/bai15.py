"""Bai 15 - Thong ke reward cua 100 episode bang NumPy.

Tinh mean, min, max, standard deviation va in voi 2 chu so thap phan.
Du lieu lay lai tu Bai 14 (cung seed nen ket qua giong het).
"""

import numpy as np

from bai14 import run_random_episodes


def compute_statistics(episode_rewards: list[float]) -> dict[str, float]:
    """Tinh cac thong ke co ban tren mang reward bang NumPy."""
    rewards = np.array(episode_rewards, dtype=np.float64)
    return {
        "mean": float(np.mean(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "std": float(np.std(rewards)),
    }


def main() -> None:
    episode_rewards, _ = run_random_episodes(n_episodes=100, seed=42)

    stats = compute_statistics(episode_rewards)

    print(f"Number of episodes : {len(episode_rewards)}")
    print()
    print(f"Mean reward : {stats['mean']:.2f}")
    print(f"Min reward  : {stats['min']:.2f}")
    print(f"Max reward  : {stats['max']:.2f}")
    print(f"Std reward  : {stats['std']:.2f}")


if __name__ == "__main__":
    main()
