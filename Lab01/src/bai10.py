"""Bai 10 - Mo rong Bai 9: cong don total_reward va in do dai episode."""

import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1")

    observation, info = env.reset(seed=42)
    env.action_space.seed(42)

    total_reward = 0.0
    episode_length = 0
    max_steps = 20

    print(" t  | action | reward | total_reward")
    print("----+--------+--------+-------------")
    for t in range(max_steps):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_length += 1

        print(f"{t:^4d}| {action:^7d}| {reward:^7.1f}| {total_reward:^12.1f}")

        if terminated or truncated:
            break

    print()
    print(f"Episode length: {episode_length}")
    print(f"Total reward  : {total_reward:.1f}")

    # Voi CartPole moi buoc song sot deu duoc reward = 1.0,
    # nen total reward luon bang do dai episode.
    print(f"Reward == length: {total_reward == float(episode_length)}")

    env.close()


if __name__ == "__main__":
    main()
