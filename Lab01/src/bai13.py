"""Bai 13 - Chay random agent trong 10 episode va in bang ket qua."""

import gymnasium as gym


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


def main() -> None:
    env = gym.make("CartPole-v1")

    # Seed mot lan o dau thi nghiem -> toan bo 10 episode tai lap duoc
    env.reset(seed=42)
    env.action_space.seed(42)

    n_episodes = 10

    print("Episode | Reward | Length")
    print("--------+--------+-------")
    for episode in range(1, n_episodes + 1):
        total_reward, episode_length = random_agent(env)
        print(f"{episode:^8d}| {total_reward:^7.1f}| {episode_length:^6d}")

    env.close()


if __name__ == "__main__":
    main()
