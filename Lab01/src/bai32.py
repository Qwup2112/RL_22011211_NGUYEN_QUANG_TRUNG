"""Bai 32 - Cai tien heuristic: dung ca goc va van toc goc cua pole.

Chi dung goc (Bai 31) khien agent phan ung tre: den luc goc doi dau thi cot da
quay rat nhanh. Ta du doan goc o tuong lai gan:

    score = pole_angle + k * pole_angular_velocity

Neu score < 0 (cot dang nghieng/quay sang trai) thi day sang trai, nguoc lai
day sang phai. Tham so k dieu chinh muc do "nhin truoc".
"""

import gymnasium as gym
import numpy as np

# He so nhin truoc: cang lon thi phan ung cang som theo van toc goc
LOOKAHEAD = 0.5


def improved_policy(observation) -> int:
    """Policy dung ca pole angle va pole angular velocity."""
    pole_angle = observation[2]
    pole_angular_velocity = observation[3]

    # Goc du doan sau mot khoang thoi gian ngan
    predicted_angle = pole_angle + LOOKAHEAD * pole_angular_velocity

    if predicted_angle < 0:
        return 0    # day sang trai
    else:
        return 1    # day sang phai


def angle_based_policy(observation) -> int:
    """Policy cua Bai 31, chi dung goc cua pole."""
    return 0 if observation[2] < 0 else 1


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
        "improved": improved_policy,
    }

    print(f"CartPole-v1, {n_episodes} episodes per policy (lookahead k = {LOOKAHEAD})")
    print()
    print("Policy      | Mean   | Std   | Min   | Max")
    print("------------+--------+-------+-------+-------")

    means = {}
    for name, policy in policies.items():
        env.reset(seed=42)
        env.action_space.seed(42)

        rewards = evaluate(env, policy, n_episodes)
        means[name] = float(rewards.mean())
        print(f"{name:<12s}| {rewards.mean():^7.2f}| {rewards.std():^6.2f}| "
              f"{rewards.min():^6.1f}| {rewards.max():^6.1f}")

    print()
    print(f"Goal: mean(improved) > mean(random)  ->  "
          f"{means['improved']:.2f} > {means['random']:.2f} = "
          f"{means['improved'] > means['random']}")
    print(f"Improved vs angle-based: {means['improved'] - means['angle_based']:+.2f}")

    env.close()


if __name__ == "__main__":
    main()
