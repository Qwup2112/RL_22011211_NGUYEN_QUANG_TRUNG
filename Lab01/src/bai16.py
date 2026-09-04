"""Bai 16 - Tim episode tot nhat tu du lieu da thu thap.

Chi dung lai mang reward/length da luu (khong chay lai moi truong de tim).
"""

import numpy as np

from bai14 import run_random_episodes


def find_best_episode(episode_rewards: list[float],
                      episode_lengths: list[int]) -> dict[str, float]:
    """Tim episode co reward lon nhat bang np.argmax tren du lieu da co."""
    rewards = np.array(episode_rewards, dtype=np.float64)

    best_index = int(np.argmax(rewards))

    return {
        "episode": best_index + 1,          # danh so episode tu 1
        "index": best_index,                # chi so trong list (tu 0)
        "reward": float(rewards[best_index]),
        "length": int(episode_lengths[best_index]),
    }


def main() -> None:
    episode_rewards, episode_lengths = run_random_episodes(n_episodes=100, seed=42)

    best = find_best_episode(episode_rewards, episode_lengths)

    print("Best episode found from the stored arrays (no extra env rollout):")
    print(f"  Episode index : {best['episode']}")
    print(f"  Reward        : {best['reward']:.2f}")
    print(f"  Length        : {best['length']}")

    # Tim them episode te nhat de doi chieu
    worst_index = int(np.argmin(np.array(episode_rewards)))
    print()
    print("Worst episode:")
    print(f"  Episode index : {worst_index + 1}")
    print(f"  Reward        : {episode_rewards[worst_index]:.2f}")
    print(f"  Length        : {episode_lengths[worst_index]}")


if __name__ == "__main__":
    main()
