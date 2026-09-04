"""Bai 3 - Kham pha action space cua CartPole.

So luong action duoc xac dinh tu chinh doi tuong action_space
(khong gan bang hang so).
"""

import gymnasium as gym
from gymnasium import spaces


def count_actions(action_space: spaces.Space) -> int:
    """Tra ve so action co the thuc hien cua mot khong gian action roi rac."""
    if isinstance(action_space, spaces.Discrete):
        # Discrete(n) -> co dung n action: 0, 1, ..., n-1
        return int(action_space.n)
    if isinstance(action_space, spaces.MultiDiscrete):
        return int(action_space.nvec.prod())
    raise TypeError(f"Khong gian action lien tuc, khong dem duoc: {action_space}")


def main() -> None:
    env = gym.make("CartPole-v1")

    print("Action space      :", env.action_space)
    print("Type              :", type(env.action_space).__name__)

    n_actions = count_actions(env.action_space)
    print(f"Number of actions: {n_actions}")

    # Liet ke toan bo action hop le
    print("Valid actions     :", list(range(n_actions)))

    # Kiem tra lai bang cach sample nhieu lan: moi gia tri phai nam trong space
    sample = env.action_space.sample()
    print("One sampled action:", sample, "-> contains:", env.action_space.contains(sample))

    env.close()


if __name__ == "__main__":
    main()
