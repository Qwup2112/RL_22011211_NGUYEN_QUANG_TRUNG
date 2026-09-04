"""Bai 34 - Ham evaluate_policy() danh gia mot policy tren nhieu episode."""

import gymnasium as gym
import numpy as np

from bai33 import make_random_policy, run_episode


def evaluate_policy(env_name: str,
                    policy,
                    n_episodes: int = 100,
                    seed: int = 42) -> dict:
    """Danh gia policy tren env_name qua n_episodes episode.

    Episode dau tien duoc reset voi seed cho truoc, cac episode sau tiep tuc
    chuoi ngau nhien do -> toan bo thi nghiem tai lap duoc.

    Tra ve dictionary gom mean/std/min/max reward va mean length.
    """
    env = gym.make(env_name)

    rewards = []
    lengths = []
    terminated_count = 0
    truncated_count = 0

    for episode_index in range(n_episodes):
        episode_seed = seed if episode_index == 0 else None
        result = run_episode(env, policy, seed=episode_seed)

        rewards.append(result["reward"])
        lengths.append(result["length"])
        terminated_count += int(result["terminated"])
        truncated_count += int(result["truncated"])

    env.close()

    rewards_array = np.array(rewards, dtype=np.float64)
    lengths_array = np.array(lengths, dtype=np.float64)

    return {
        "env_name": env_name,
        "n_episodes": n_episodes,
        "seed": seed,
        "mean_reward": float(rewards_array.mean()),
        "std_reward": float(rewards_array.std()),
        "min_reward": float(rewards_array.min()),
        "max_reward": float(rewards_array.max()),
        "mean_length": float(lengths_array.mean()),
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "rewards": rewards,
        "lengths": lengths,
    }


def main() -> None:
    n_episodes = 100

    for env_name in ["CartPole-v1", "FrozenLake-v1"]:
        policy = make_random_policy(env_name, seed=42)
        result = evaluate_policy(env_name, policy, n_episodes=n_episodes, seed=42)

        print(f"=== {env_name} - random policy, {n_episodes} episodes ===")
        print(f"mean_reward : {result['mean_reward']:.2f}")
        print(f"std_reward  : {result['std_reward']:.2f}")
        print(f"min_reward  : {result['min_reward']:.2f}")
        print(f"max_reward  : {result['max_reward']:.2f}")
        print(f"mean_length : {result['mean_length']:.2f}")
        print(f"terminated  : {result['terminated_count']}, "
              f"truncated: {result['truncated_count']}")
        print()


if __name__ == "__main__":
    main()
