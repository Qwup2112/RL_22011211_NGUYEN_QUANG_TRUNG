"""Bai 29 - Viet policy duoi dang ham.

Agent khong goi truc tiep env.action_space.sample() nua ma luon goi
policy(observation). Nho vay chi can thay ham policy la doi duoc agent.
"""

import gymnasium as gym
import numpy as np


def make_random_policy(env: gym.Env):
    """Tra ve mot ham policy(observation) chon action ngau nhien tu env."""

    def policy(observation):
        # Ban dau policy chi tra ve action ngau nhien, chua dung den observation
        return env.action_space.sample()

    return policy


def run_episode_with_policy(env: gym.Env, policy, max_steps: int = 500):
    """Chay mot episode, moi buoc chon action bang policy(observation)."""
    observation, info = env.reset()

    total_reward = 0.0
    length = 0

    for _ in range(max_steps):
        action = policy(observation)          # thay cho env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        length += 1

        if terminated or truncated:
            break

    return total_reward, length


def main() -> None:
    env = gym.make("CartPole-v1")

    env.reset(seed=42)
    env.action_space.seed(42)

    policy = make_random_policy(env)

    rewards = []
    for episode in range(10):
        total_reward, length = run_episode_with_policy(env, policy)
        rewards.append(total_reward)
        print(f"Episode {episode + 1:2d}: reward={total_reward:6.1f}, length={length:3d}")

    print()
    print(f"Mean reward over {len(rewards)} episodes: {np.mean(rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()
