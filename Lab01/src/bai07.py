"""Bai 7 - Mot buoc tuong tac duy nhat voi CartPole.

In day du: trang thai truoc, action, trang thai sau, reward, terminated,
truncated va info.
"""

import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1")

    # reset() luon phai duoc goi truoc step() dau tien
    state_before, reset_info = env.reset(seed=42)

    env.action_space.seed(42)
    action = env.action_space.sample()

    # API moi tra ve 5 gia tri (khong con bien done nhu Gym cu)
    state_after, reward, terminated, truncated, info = env.step(action)

    print("State before action:", state_before)
    print("Action             :", action)
    print("State after action :", state_after)
    print("Reward             :", reward)
    print("Terminated         :", terminated)
    print("Truncated          :", truncated)
    print("Info               :", info)

    print()
    print("Reset info         :", reset_info)
    print("Delta state        :", state_after - state_before)

    env.close()


if __name__ == "__main__":
    main()
