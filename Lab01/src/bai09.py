"""Bai 9 - Chay toi da 20 timestep voi action ngau nhien.

Moi timestep in t, action, reward; dung ngay khi terminated hoac truncated.
"""

import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1")

    observation, info = env.reset(seed=42)
    env.action_space.seed(42)

    max_steps = 20

    print(" t  | action | reward")
    print("----+--------+-------")
    for t in range(max_steps):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        print(f"{t:^4d}| {action:^7d}| {reward:^6.1f}")

        if terminated or truncated:
            print(f"Episode ended at t={t} (terminated={terminated}, truncated={truncated}).")
            break
    else:
        print(f"Reached the limit of {max_steps} timesteps without ending.")

    env.close()


if __name__ == "__main__":
    main()
