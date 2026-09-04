"""Chuong trinh tong hop cua Lab01 (hoan thien khung o muc 9 cua de bai).

Chay thi nghiem tren CartPole-v1 voi hai policy:
    - random_policy      : chon action ngau nhien
    - heuristic_policy   : dung pole angle + pole angular velocity

Ket qua: bang thong ke, bieu do reward va moving average trong Lab01/figures/.

Cach chay:
    cd Lab01
    python src/main.py
"""

from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ENV_NAME = "CartPole-v1"
N_EPISODES = 500
SEED = 42
WINDOW_SIZE = 20

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def random_policy(observation, env):
    """Policy ngau nhien: bo qua observation, sample tu action space."""
    return env.action_space.sample()


def heuristic_policy(observation):
    """Policy heuristic: du doan goc pole sap toi roi day xe ve phia do."""
    predicted_angle = observation[2] + 0.5 * observation[3]
    return 0 if predicted_angle < 0 else 1


def run_episode(env, policy, seed=None, max_steps=1000):
    """Chay mot episode; tra ve dict reward / length / terminated / truncated."""
    if seed is None:
        observation, info = env.reset()
    else:
        observation, info = env.reset(seed=seed)

    total_reward = 0.0
    length = 0
    terminated = False
    truncated = False

    while length < max_steps:
        action = policy(observation)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        length += 1

        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "length": length,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def evaluate_policy(env_name, policy_factory, n_episodes=100, seed=42):
    """Danh gia mot policy tren env_name qua n_episodes episode.

    policy_factory nhan env va tra ve ham policy(observation), nho vay ham nay
    dung duoc ca voi random policy (can env) lan heuristic policy.
    """
    env = gym.make(env_name)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    policy = policy_factory(env)

    rewards = []
    lengths = []
    terminated_count = 0
    truncated_count = 0

    for _ in range(n_episodes):
        result = run_episode(env, policy)

        rewards.append(result["reward"])
        lengths.append(result["length"])
        terminated_count += int(result["terminated"])
        truncated_count += int(result["truncated"])

    env.close()      # luon dong moi truong sau khi thi nghiem xong

    rewards_array = np.array(rewards, dtype=np.float64)
    best_index = int(np.argmax(rewards_array))
    worst_index = int(np.argmin(rewards_array))

    return {
        "env_name": env_name,
        "rewards": rewards,
        "lengths": lengths,
        "mean_reward": float(rewards_array.mean()),
        "std_reward": float(rewards_array.std()),
        "min_reward": float(rewards_array.min()),
        "max_reward": float(rewards_array.max()),
        "mean_length": float(np.mean(lengths)),
        "best_episode": best_index + 1,
        "worst_episode": worst_index + 1,
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
    }


def moving_average(values, window_size):
    """Trung binh truot tu cai dat (khong dung pandas.rolling)."""
    averages = []
    running_sum = float(sum(values[:window_size]))
    averages.append(running_sum / window_size)

    for index in range(window_size, len(values)):
        running_sum += float(values[index]) - float(values[index - window_size])
        averages.append(running_sum / window_size)

    return averages


def plot_rewards(results, output_path):
    """Ve reward theo episode va moving average cho tung agent."""
    figure, axes = plt.subplots(2, 1, figsize=(11, 8))

    colors = {"random": "#8c8c8c", "heuristic": "#1f77b4"}

    for name, result in results.items():
        episodes = np.arange(1, len(result["rewards"]) + 1)
        axes[0].plot(episodes, result["rewards"], linewidth=0.9,
                     color=colors[name], alpha=0.8, label=name)

        smoothed = moving_average(result["rewards"], WINDOW_SIZE)
        smoothed_x = np.arange(WINDOW_SIZE, len(result["rewards"]) + 1)
        axes[1].plot(smoothed_x, smoothed, linewidth=2.0,
                     color=colors[name], label=f"{name} (window={WINDOW_SIZE})")

    axes[0].set_title(f"{ENV_NAME}: reward per episode (seed = {SEED})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].grid(True, linestyle=":", alpha=0.7)
    axes[0].legend()

    axes[1].set_title("Moving average of the reward")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Moving average reward")
    axes[1].grid(True, linestyle=":", alpha=0.7)
    axes[1].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    print("=" * 72)
    print(f"Lab01 - summary program - {ENV_NAME} - {N_EPISODES} episodes")
    print("=" * 72)
    print()

    # 1) Tao va chay thi nghiem cho tung agent
    results = {
        "random": evaluate_policy(
            ENV_NAME,
            lambda env: (lambda observation: random_policy(observation, env)),
            n_episodes=N_EPISODES,
            seed=SEED,
        ),
        "heuristic": evaluate_policy(
            ENV_NAME,
            lambda env: heuristic_policy,
            n_episodes=N_EPISODES,
            seed=SEED,
        ),
    }

    # 2) In thong ke
    print("Agent     | Mean   | Std    | Min   | Max   | Mean len | term | trunc")
    print("----------+--------+--------+-------+-------+----------+------+------")
    for name, result in results.items():
        print(f"{name:<10s}| {result['mean_reward']:^7.2f}| {result['std_reward']:^7.2f}| "
              f"{result['min_reward']:^6.1f}| {result['max_reward']:^6.1f}| "
              f"{result['mean_length']:^9.2f}| {result['terminated_count']:^5d}| "
              f"{result['truncated_count']:^5d}")

    print()
    for name, result in results.items():
        print(f"{name}: best episode #{result['best_episode']} "
              f"(reward {max(result['rewards']):.1f}), "
              f"worst episode #{result['worst_episode']} "
              f"(reward {min(result['rewards']):.1f})")

    # 3) Ve bieu do
    output_path = FIGURES_DIR / "main_summary.png"
    plot_rewards(results, output_path)
    print()
    print(f"Figure saved to: {output_path}")

    # 4) Ket luan
    gain = results["heuristic"]["mean_reward"] - results["random"]["mean_reward"]
    print()
    print("Conclusion:")
    print(f"  The heuristic policy gains {gain:+.2f} reward per episode over random.")
    print("  Both agents are fixed rules: the moving average stays flat because")
    print("  nothing is learned from the reward signal. This is the baseline that")
    print("  real RL algorithms (Q-Learning, SARSA, DQN) will have to beat.")
    print()
    print("For the full mini-project of exercise 36, run: python src/bai36.py")


if __name__ == "__main__":
    main()
