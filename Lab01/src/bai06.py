"""Bai 6 - Sinh 20 action ngau nhien, luu vao list va thong ke tan suat."""

from collections import Counter

import gymnasium as gym


def sample_actions(env: gym.Env, n_actions: int) -> list[int]:
    """Sinh n_actions action ngau nhien tu action_space va tra ve mot list."""
    return [int(env.action_space.sample()) for _ in range(n_actions)]


def main() -> None:
    env = gym.make("CartPole-v1")

    # Seed action space de ket qua tai lap duoc giua cac lan chay
    env.action_space.seed(42)

    n_samples = 20
    actions = sample_actions(env, n_samples)

    print(f"Sampled {n_samples} random actions:")
    print(actions)

    # Dem tan suat xuat hien cua tung action
    counter = Counter(actions)

    print()
    print("Action | Count | Frequency")
    print("-------+-------+----------")
    for action in range(int(env.action_space.n)):
        count = counter.get(action, 0)
        print(f"{action:^6d} | {count:^5d} | {count / n_samples:8.2%}")

    env.close()


if __name__ == "__main__":
    main()
