"""Bai 36 - Mini-project: pipeline thi nghiem Agent-Environment hoan chinh.

Moi truong duoc chon: CartPole-v1.

Cau truc chuong trinh theo dung yeu cau cua de bai:
    create_environment()  - tao moi truong
    policy()              - quy tac chon action tu observation
    run_episode()         - chay mot episode
    evaluate_policy()     - danh gia policy tren nhieu episode
    plot_results()        - ve bieu do reward va moving average
    main()                - dieu phoi toan bo thi nghiem

Toan bo chuong trinh dung API Gymnasium moi (terminated / truncated), co seed
de tai lap va luon goi env.close() khi ket thuc.
"""

from pathlib import Path

import gymnasium as gym
import matplotlib

# Backend "Agg" giup luu hinh duoc ma khong can cua so do hoa
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ENV_NAME = "CartPole-v1"
N_EPISODES = 500
SEED = 42
WINDOW_SIZE = 10
LOOKAHEAD = 0.5

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"


def create_environment(env_name: str = ENV_NAME, seed=None) -> gym.Env:
    """Tao moi truong Gymnasium va seed neu can."""
    env = gym.make(env_name)

    if seed is not None:
        env.reset(seed=seed)          # seed cho bo sinh so cua moi truong
        env.action_space.seed(seed)   # seed rieng cho action space

    return env


def policy(observation) -> int:
    """Heuristic policy cho CartPole: dung goc va van toc goc cua pole.

    observation[2] = pole angle, observation[3] = pole angular velocity.
    Du doan goc sap toi roi day xe ve phia do de giu cot thang dung.
    """
    predicted_angle = observation[2] + LOOKAHEAD * observation[3]
    return 0 if predicted_angle < 0 else 1


def make_random_policy(env: gym.Env):
    """Policy ngau nhien, dung lam baseline so sanh."""

    def random_policy(observation):
        return env.action_space.sample()

    return random_policy


def run_episode(env: gym.Env, policy_fn, seed=None, max_steps: int = 1000) -> dict:
    """Chay mot episode va tra ve reward, length, terminated, truncated."""
    if seed is None:
        observation, info = env.reset()
    else:
        observation, info = env.reset(seed=seed)

    total_reward = 0.0
    length = 0
    terminated = False
    truncated = False

    while length < max_steps:
        action = policy_fn(observation)

        # API moi: 5 gia tri tra ve, khong dung bien done cua Gym cu
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


def evaluate_policy(env: gym.Env, policy_fn, n_episodes: int = N_EPISODES) -> dict:
    """Chay policy trong n_episodes episode va tong hop thong ke."""
    rewards = []
    lengths = []
    terminated_count = 0
    truncated_count = 0

    for _ in range(n_episodes):
        result = run_episode(env, policy_fn)

        rewards.append(result["reward"])
        lengths.append(result["length"])
        terminated_count += int(result["terminated"])
        truncated_count += int(result["truncated"])

    rewards_array = np.array(rewards, dtype=np.float64)
    lengths_array = np.array(lengths, dtype=np.float64)

    # np.argmax / np.argmin lam viec tren du lieu da luu, khong chay lai env
    best_index = int(np.argmax(rewards_array))
    worst_index = int(np.argmin(rewards_array))

    return {
        "rewards": rewards,
        "lengths": lengths,
        "mean_reward": float(rewards_array.mean()),
        "std_reward": float(rewards_array.std()),
        "min_reward": float(rewards_array.min()),
        "max_reward": float(rewards_array.max()),
        "mean_length": float(lengths_array.mean()),
        "best_episode": {
            "episode": best_index + 1,
            "reward": float(rewards_array[best_index]),
            "length": int(lengths[best_index]),
        },
        "worst_episode": {
            "episode": worst_index + 1,
            "reward": float(rewards_array[worst_index]),
            "length": int(lengths[worst_index]),
        },
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
    }


def moving_average(values, window_size: int = WINDOW_SIZE) -> list[float]:
    """Trung binh truot tu cai dat (khong dung pandas)."""
    if window_size <= 0 or window_size > len(values):
        raise ValueError("window_size khong hop le")

    averages = []
    running_sum = float(sum(values[:window_size]))
    averages.append(running_sum / window_size)

    for index in range(window_size, len(values)):
        running_sum += float(values[index]) - float(values[index - window_size])
        averages.append(running_sum / window_size)

    return averages


def plot_results(rewards: list[float],
                 lengths: list[int],
                 window_size: int = WINDOW_SIZE,
                 baseline_rewards=None,
                 baseline_label: str = "random policy") -> list[Path]:
    """Ve bieu do reward theo episode va bieu do moving average.

    baseline_rewards (neu co) duoc ve chong len de so sanh voi agent chinh.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(rewards) + 1)
    saved_paths = []

    # --- Bieu do 1: reward va episode length theo episode ---
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(episodes, rewards, linewidth=1.0, color="#1f77b4",
                 label="heuristic policy")
    axes[0].axhline(float(np.mean(rewards)), color="#d62728", linestyle="--",
                    label=f"Mean = {np.mean(rewards):.2f}")
    if baseline_rewards is not None:
        axes[0].plot(episodes, baseline_rewards, linewidth=0.9, color="#8c8c8c",
                     alpha=0.9, label=baseline_label)
    axes[0].set_title(f"Mini-project on {ENV_NAME}: reward per episode "
                      f"({len(rewards)} episodes, seed = {SEED})")
    axes[0].set_ylabel("Total reward")
    axes[0].grid(True, linestyle=":", alpha=0.7)
    axes[0].legend()

    axes[1].plot(episodes, lengths, linewidth=1.0, color="#2ca02c")
    axes[1].set_title("Episode length per episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode length")
    axes[1].grid(True, linestyle=":", alpha=0.7)

    figure.tight_layout()
    reward_path = FIGURES_DIR / "mini_project_rewards.png"
    figure.savefig(reward_path, dpi=150)
    plt.close(figure)
    saved_paths.append(reward_path)

    # --- Bieu do 2: reward goc va moving average ---
    smoothed = moving_average(rewards, window_size)
    smoothed_x = np.arange(window_size, len(rewards) + 1)

    figure, axes = plt.subplots(figsize=(11, 5))
    axes.plot(episodes, rewards, color="#aec7e8", linewidth=0.9, label="Raw reward")
    axes.plot(smoothed_x, smoothed, color="#d62728", linewidth=2.0,
              label=f"Moving average (window = {window_size})")
    if baseline_rewards is not None:
        baseline_smoothed = moving_average(baseline_rewards, window_size)
        axes.plot(episodes, baseline_rewards, color="#d9d9d9", linewidth=0.8,
                  label=f"Raw reward - {baseline_label}")
        axes.plot(smoothed_x, baseline_smoothed, color="#4d4d4d", linewidth=2.0,
                  label=f"Moving average - {baseline_label}")
    axes.set_title(f"Mini-project on {ENV_NAME}: reward and moving average")
    axes.set_xlabel("Episode")
    axes.set_ylabel("Total reward")
    axes.grid(True, linestyle=":", alpha=0.7)
    axes.legend()

    figure.tight_layout()
    moving_path = FIGURES_DIR / "mini_project_moving_average.png"
    figure.savefig(moving_path, dpi=150)
    plt.close(figure)
    saved_paths.append(moving_path)

    return saved_paths


def save_experiment_data(rewards: list[float], lengths: list[int]) -> Path:
    """Luu reward va length cua tung episode ra file CSV trong data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "mini_project_episodes.csv"

    with output_path.open("w", encoding="utf-8") as file:
        file.write("episode,reward,length\n")
        for index, (reward, length) in enumerate(zip(rewards, lengths), start=1):
            file.write(f"{index},{reward},{length}\n")

    return output_path


def print_report(title: str, result: dict) -> None:
    """In bang thong ke cua mot agent."""
    print(f"--- {title} ---")
    print(f"  Episodes      : {len(result['rewards'])}")
    print(f"  Mean reward   : {result['mean_reward']:.2f}")
    print(f"  Std reward    : {result['std_reward']:.2f}")
    print(f"  Min reward    : {result['min_reward']:.2f}")
    print(f"  Max reward    : {result['max_reward']:.2f}")
    print(f"  Mean length   : {result['mean_length']:.2f}")
    print(f"  Best episode  : #{result['best_episode']['episode']} "
          f"(reward {result['best_episode']['reward']:.1f}, "
          f"length {result['best_episode']['length']})")
    print(f"  Worst episode : #{result['worst_episode']['episode']} "
          f"(reward {result['worst_episode']['reward']:.1f}, "
          f"length {result['worst_episode']['length']})")
    print(f"  terminated={result['terminated_count']}, "
          f"truncated={result['truncated_count']}")
    print()


def main() -> None:
    print("=" * 68)
    print(f"MINI-PROJECT - {ENV_NAME} - {N_EPISODES} episodes - seed {SEED}")
    print("=" * 68)
    print()

    # 1) Baseline: random policy
    random_env = create_environment(ENV_NAME, seed=SEED)
    random_result = evaluate_policy(random_env, make_random_policy(random_env),
                                    n_episodes=N_EPISODES)
    random_env.close()
    print_report("Agent 1: random policy", random_result)

    # 2) Agent chinh: heuristic policy
    env = create_environment(ENV_NAME, seed=SEED)
    result = evaluate_policy(env, policy, n_episodes=N_EPISODES)
    env.close()
    print_report("Agent 2: heuristic policy (angle + angular velocity)", result)

    # 3) Luu du lieu va ve bieu do
    data_path = save_experiment_data(result["rewards"], result["lengths"])
    figure_paths = plot_results(result["rewards"], result["lengths"],
                                baseline_rewards=random_result["rewards"])

    print(f"Episode data saved to : {data_path}")
    for path in figure_paths:
        print(f"Figure saved to       : {path}")

    # 4) Ket luan rut ra tu ket qua thi nghiem
    improvement = result["mean_reward"] - random_result["mean_reward"]
    ratio = result["mean_reward"] / random_result["mean_reward"]

    print()
    print("CONCLUSION")
    print("-" * 68)
    print(f"1. The heuristic agent gains {improvement:+.2f} reward per episode over")
    print(f"   the random agent ({ratio:.1f}x better mean reward).")
    print(f"2. The random agent ends with terminated=True in "
          f"{random_result['terminated_count']}/{N_EPISODES} episodes (the pole falls)")
    print(f"   after {random_result['mean_length']:.1f} steps on average, while the "
          f"heuristic agent ends")
    print(f"   with truncated=True in {result['truncated_count']}/{N_EPISODES} episodes: "
          f"it hits the 500-step")
    print("   TimeLimit, which is the best possible outcome on CartPole-v1.")
    print(f"3. Std goes from {random_result['std_reward']:.2f} (random) to "
          f"{result['std_reward']:.2f} (heuristic): the rule works for")
    print("   every random initial state produced by reset(), it is not luck.")
    print("4. The moving average curve is flat: neither agent learns, they only")
    print("   apply a fixed rule. A real RL algorithm would show an upward trend.")
    print("5. Seeding makes the whole experiment reproducible: running this script")
    print("   twice gives exactly the same numbers.")


if __name__ == "__main__":
    main()
