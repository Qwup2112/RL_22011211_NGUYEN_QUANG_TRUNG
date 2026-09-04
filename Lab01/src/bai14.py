"""Bai 14 - Chay random agent trong 100 episode, luu reward vao list.

Khong in tung timestep. Ket qua duoc luu them ra thu muc data/ de cac bai sau
co the doi chieu.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def random_agent(env: gym.Env, max_steps: int = 500) -> tuple[float, int]:
    """Chay mot episode voi action ngau nhien, tra ve (total_reward, length)."""
    observation, info = env.reset()

    total_reward = 0.0
    episode_length = 0

    for _ in range(max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        episode_length += 1

        if terminated or truncated:
            break

    return total_reward, episode_length


def run_random_episodes(env_name: str = "CartPole-v1",
                        n_episodes: int = 100,
                        seed: int = 42) -> tuple[list[float], list[int]]:
    """Chay n_episodes episode bang random agent, tra ve (rewards, lengths)."""
    env = gym.make(env_name)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        total_reward, episode_length = random_agent(env)
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)

    env.close()
    return episode_rewards, episode_lengths


def main() -> None:
    n_episodes = 100
    episode_rewards, episode_lengths = run_random_episodes(n_episodes=n_episodes)

    print(f"Ran {n_episodes} episodes with a random agent on CartPole-v1.")
    print(f"len(episode_rewards) = {len(episode_rewards)}")
    print()
    print("First 10 rewards:", episode_rewards[:10])
    print("Last 10 rewards :", episode_rewards[-10:])
    print()
    print(f"Sum of all rewards: {sum(episode_rewards):.1f}")

    # Luu lai du lieu thi nghiem de dung cho cac bai 15-18
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "random_agent_100_episodes.csv"
    with output_path.open("w", encoding="utf-8") as file:
        file.write("episode,reward,length\n")
        for index, (reward, length) in enumerate(zip(episode_rewards, episode_lengths), start=1):
            file.write(f"{index},{reward},{length}\n")

    print(f"Saved experiment data to: {output_path}")
    print("Mean reward (quick check):", f"{np.mean(episode_rewards):.2f}")


if __name__ == "__main__":
    main()
