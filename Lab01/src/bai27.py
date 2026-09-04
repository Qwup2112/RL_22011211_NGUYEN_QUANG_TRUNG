"""Bai 27 - Reward trong FrozenLake voi random policy.

Chay 200 episode (nhieu hon muc toi thieu 100), dem so lan thanh cong /
that bai va tinh success rate.
"""

import gymnasium as gym
import numpy as np


def run_random_episodes(env: gym.Env, n_episodes: int):
    """Chay n_episodes episode bang random policy.

    Tra ve (rewards, lengths) cua tung episode.
    """
    rewards = []
    lengths = []

    for _ in range(n_episodes):
        observation, info = env.reset()
        total_reward = 0.0
        length = 0

        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            length += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(length)

    return rewards, lengths


def main() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)

    env.reset(seed=42)
    env.action_space.seed(42)

    total_episodes = 200
    rewards, lengths = run_random_episodes(env, total_episodes)

    # Trong FrozenLake, reward = 1 chi khi toi Goal, moi truong hop khac = 0
    success = int(sum(1 for reward in rewards if reward > 0))
    failure = total_episodes - success

    success_rate = success / total_episodes

    print("FrozenLake-v1 (is_slippery=False) with a random policy")
    print(f"Total episodes : {total_episodes}")
    print(f"Success        : {success}")
    print(f"Failure        : {failure}")
    print(f"Success rate   : {success_rate:.4f}  ({success_rate:.2%})")
    print()
    print(f"Average reward : {np.mean(rewards):.4f}")
    print(f"Average length : {np.mean(lengths):.2f}")
    print()
    print("Note: mean reward equals the success rate because reward is 1 only")
    print("when the agent reaches the Goal and 0 in every other case.")

    env.close()


if __name__ == "__main__":
    main()
