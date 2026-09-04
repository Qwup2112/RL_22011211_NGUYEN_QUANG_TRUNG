"""Bai 26 - Dieu khien FrozenLake bang mot chuoi action tu thiet ke.

Voi is_slippery=False moi truong la tat dinh nen co the di tu Start toi Goal
bang mot chuoi action co dinh.

Ban do 4x4 mac dinh:
    S F F F      0  1  2  3
    F H F H      4  5  6  7
    F F F H      8  9 10 11
    H F F G     12 13 14 15

Duong di duoc chon: 0 -> 4 -> 8 -> 9 -> 10 -> 14 -> 15
"""

import gymnasium as gym

ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

# DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT
actions = [1, 1, 2, 2, 1, 2]


def main() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")

    observation, info = env.reset(seed=42)
    print(f"Start state: {observation}")
    print(env.render())

    total_reward = 0.0

    for step_index, action in enumerate(actions, start=1):
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        row, col = divmod(int(observation), env.unwrapped.desc.shape[1])
        print(f"Step {step_index}: action={action} ({ACTION_NAMES[action]}) -> "
              f"state={observation} (row {row}, col {col}), reward={reward}, "
              f"terminated={terminated}, truncated={truncated}")
        print(env.render())

        if terminated or truncated:
            break

    print(f"Total reward: {total_reward}")
    if total_reward > 0:
        print("The agent reached the Goal.")
    else:
        print("The agent did NOT reach the Goal.")

    env.close()


if __name__ == "__main__":
    main()
