"""Bai 11 - Random agent hoan chinh cho mot episode."""

import gymnasium as gym


def random_agent(env: gym.Env, max_steps: int = 500) -> tuple[float, int]:
    """Chay mot episode voi action ngau nhien.

    Cac buoc: reset moi truong, chon action ngau nhien, tuong tac cho den khi
    episode ket thuc (hoac cham max_steps).

    Tra ve (total_reward, episode_length).
    """
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

    # Seed de ket qua tai lap duoc
    env.reset(seed=42)
    env.action_space.seed(42)

    for episode in range(5):
        total_reward, episode_length = random_agent(env, max_steps=500)
        print(f"Episode {episode + 1}: total_reward={total_reward:6.1f}, "
              f"episode_length={episode_length:3d}")

    env.close()


if __name__ == "__main__":
    main()
