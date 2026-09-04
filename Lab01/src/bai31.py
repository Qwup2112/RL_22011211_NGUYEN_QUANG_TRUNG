"""Bai 31 - Heuristic policy dua tren goc cua pole (CartPole).

Observation cua CartPole-v1:
    observation[0] = cart position          (vi tri xe)
    observation[1] = cart velocity          (van toc xe)
    observation[2] = pole angle (radian)    (goc nghieng cua cot)
    observation[3] = pole angular velocity  (van toc goc cua cot)

Y tuong: cot nghieng ve phia nao thi day xe ve phia do de "do" lai cot.
"""

import gymnasium as gym
import numpy as np


def angle_based_policy(observation) -> int:
    """Chon action dua tren dau cua goc pole."""
    pole_angle = observation[2]

    if pole_angle < 0:
        # Cot nghieng sang trai -> day xe sang trai de dua diem tua ve duoi cot
        return 0
    else:
        # Cot nghieng sang phai -> day xe sang phai
        return 1


def random_policy_factory(env: gym.Env):
    """Policy ngau nhien dung lam moc so sanh."""

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
        "random": random_policy_factory(env),
        "angle_based": angle_based_policy,
    }

    print(f"CartPole-v1, {n_episodes} episodes per policy")
    print()
    print("Policy      | Mean   | Std   | Min  | Max")
    print("------------+--------+-------+------+------")

    means = {}
    for name, policy in policies.items():
        env.reset(seed=42)
        env.action_space.seed(42)

        rewards = evaluate(env, policy, n_episodes)
        means[name] = rewards.mean()
        print(f"{name:<12s}| {rewards.mean():^7.2f}| {rewards.std():^6.2f}| "
              f"{rewards.min():^5.1f}| {rewards.max():^5.1f}")

    print()
    improvement = means["angle_based"] - means["random"]
    print(f"Angle-based minus random: {improvement:+.2f} reward per episode")
    print(f"Angle-based is better: {means['angle_based'] > means['random']}")

    env.close()


if __name__ == "__main__":
    main()
