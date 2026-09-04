"""Chuong trinh khoi dong (muc 6 cua de bai).

Tao CartPole-v1, in thong tin cac khong gian, chay toi da 100 buoc voi
action ngau nhien roi dong moi truong.
"""

import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1")

    observation, info = env.reset(seed=42)

    print("Initial observation:", observation)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    for t in range(100):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={t:3d}, "
            f"action={action}, "
            f"reward={reward}, "
            f"terminated={terminated}, "
            f"truncated={truncated}"
        )

        if terminated or truncated:
            print("Episode ended.")
            break

    env.close()


if __name__ == "__main__":
    main()
