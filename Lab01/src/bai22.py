"""Bai 22 - Ham thi nghiem co seed: experiment(seed, n_episodes)."""

import gymnasium as gym
import numpy as np


def experiment(seed: int, n_episodes: int) -> dict:
    """Chay random agent tren CartPole-v1 voi mot seed va tra ve thong ke."""
    env = gym.make("CartPole-v1")

    # Seed ca moi truong lan action space de thi nghiem tai lap duoc
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

    rewards_array = np.array(rewards, dtype=np.float64)
    return {
        "seed": seed,
        "mean_reward": float(rewards_array.mean()),
        "std_reward": float(rewards_array.std()),
        "max_reward": float(rewards_array.max()),
        "min_reward": float(rewards_array.min()),
    }


def main() -> None:
    seeds = [0, 42, 100, 2024, 31337, 7]
    n_episodes = 50

    print(f"Random agent on CartPole-v1, {n_episodes} episodes per seed")
    print()
    print(" Seed | Mean   | Std    | Min    | Max")
    print("------+--------+--------+--------+--------")

    all_results = []
    for seed in seeds:
        result = experiment(seed, n_episodes)
        all_results.append(result)
        print(f"{result['seed']:^6d}| {result['mean_reward']:^7.2f}| "
              f"{result['std_reward']:^7.2f}| {result['min_reward']:^7.2f}| "
              f"{result['max_reward']:^7.2f}")

    means = np.array([result["mean_reward"] for result in all_results])
    print()
    print(f"Mean over {len(seeds)} seeds : {means.mean():.2f}")
    print(f"Std of the seed means : {means.std():.2f}")
    print()
    print("Full dictionary of the first experiment:")
    print(all_results[0])


if __name__ == "__main__":
    main()
