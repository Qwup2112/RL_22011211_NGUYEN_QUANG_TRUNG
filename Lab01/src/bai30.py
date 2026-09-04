"""Bai 30 - Policy luon chon mot action co dinh tren CartPole.

CartPole co 2 action: 0 = day xe sang TRAI, 1 = day xe sang PHAI.
"""

import gymnasium as gym
import numpy as np


def always_left_policy(observation) -> int:
    """Luon day xe sang trai."""
    return 0


def always_right_policy(observation) -> int:
    """Luon day xe sang phai."""
    return 1


def random_policy_factory(env: gym.Env):
    """Policy ngau nhien, dung lam moc so sanh."""

    def policy(observation):
        return env.action_space.sample()

    return policy


def evaluate(env: gym.Env, policy, n_episodes: int = 100) -> np.ndarray:
    """Chay policy trong n_episodes episode, tra ve mang reward."""
    rewards = []

    for _ in range(n_episodes):
        observation, info = env.reset()
        total_reward = 0.0

        while True:
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        rewards.append(total_reward)

    return np.array(rewards)


def main() -> None:
    n_episodes = 100
    env = gym.make("CartPole-v1")

    policies = {
        "always_left": always_left_policy,
        "always_right": always_right_policy,
        "random": random_policy_factory(env),
    }

    print(f"CartPole-v1, {n_episodes} episodes per policy")
    print()
    print("Policy       | Mean  | Std  | Min  | Max")
    print("-------------+-------+------+------+------")

    means = {}

    for name, policy in policies.items():
        # Seed lai truoc moi policy de cac policy duoc so sanh cong bang
        env.reset(seed=42)
        env.action_space.seed(42)

        rewards = evaluate(env, policy, n_episodes)
        means[name] = float(rewards.mean())
        print(f"{name:<13s}| {rewards.mean():^6.2f}| {rewards.std():^5.2f}| "
              f"{rewards.min():^5.1f}| {rewards.max():^5.1f}")

    print()
    ratio = means["random"] / max(means["always_left"], means["always_right"])
    print("Nhan xet:")
    print(f"- always_left  mean = {means['always_left']:.2f}")
    print(f"- always_right mean = {means['always_right']:.2f}")
    print(f"- random       mean = {means['random']:.2f}  "
          f"({ratio:.1f}x tot hon policy co dinh tot nhat)")
    print("Hai policy co dinh chi day xe ve mot phia nen cot nga rat nhanh;")
    print("random policy doi huong lien tuc nen vo tinh giu duoc cot lau hon.")

    env.close()


if __name__ == "__main__":
    main()
