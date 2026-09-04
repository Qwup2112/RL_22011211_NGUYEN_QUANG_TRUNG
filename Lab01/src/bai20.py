"""Bai 20 - So sanh hai seed 42 va 100, moi seed chay 20 episode."""

import gymnasium as gym
import numpy as np


def run_random_episodes(env_name: str, n_episodes: int, seed: int) -> list[float]:
    """Chay n_episodes episode bang random agent voi mot seed cho truoc."""
    env = gym.make(env_name)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    rewards = []
    for _ in range(n_episodes):
        observation, info = env.reset()
        total_reward = 0.0

        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()
    return rewards


def main() -> None:
    n_episodes = 20
    seeds = [42, 100]

    results = {}
    for seed in seeds:
        rewards = run_random_episodes("CartPole-v1", n_episodes, seed)
        results[seed] = rewards

    print(f"Random agent on CartPole-v1, {n_episodes} episodes per seed")
    print()
    print("Seed | Mean reward | Std   | Min  | Max")
    print("-----+-------------+-------+------+------")
    for seed, rewards in results.items():
        array = np.array(rewards)
        print(f"{seed:^5d}| {array.mean():^12.2f}| {array.std():^6.2f}| "
              f"{array.min():^5.1f}| {array.max():^5.1f}")

    difference = np.mean(results[42]) - np.mean(results[100])
    print()
    print(f"Mean(seed=42) - Mean(seed=100) = {difference:.2f}")

    # Hai seed cho hai chuoi so ngau nhien khac nhau nen reward trung binh
    # khac nhau, nhung deu quanh muc ~20-25 cua mot random policy tren CartPole.
    # Chenh lech nay la nhieu thong ke, khong phai do agent "gioi" hon.


if __name__ == "__main__":
    main()
