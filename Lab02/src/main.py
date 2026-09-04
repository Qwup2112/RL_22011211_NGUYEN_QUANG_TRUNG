"""Chuong trinh tong hop cua Lab02 - hoan thien khung o muc 11 cua de bai.

Giai FrozenLake-v1 bang Dynamic Programming (Value Iteration va Policy
Iteration), in value table + policy dang luoi, danh gia bang mo phong va ve
duong hoi tu.

Cach chay:
    cd Lab02
    python src/main.py                          # mac dinh 4x4, is_slippery=True
    python src/main.py --gamma 0.9 --theta 1e-6
    python src/main.py --map-name 8x8 --no-slippery
    python src/main.py --all                    # chay ca hai cau hinh (Bai 36)

Cac ham thuat toan duoc cai dat trong src/mdp_utils.py va import lai o day
theo dung yeu cau muc 14; khong dung thu vien RL co san thuat toan DP.
"""

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mdp_utils import (ACTION_SYMBOLS, create_environment,  # noqa: E402
                       env_sizes, evaluate_policy_by_simulation,
                       get_transition_model, greedy_policy_from_value,
                       policy_evaluation, policy_iteration, print_policy,
                       print_value_grid, q_from_v, uniform_random_policy,
                       value_iteration)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def plot_convergence(deltas, theta: float, title: str, output_path: Path) -> Path:
    """Ve delta cua Value Iteration theo iteration."""
    figure, axes = plt.subplots(figsize=(10, 5.5))
    axes.plot(range(1, len(deltas) + 1), deltas, linewidth=2.0, color="#1f77b4",
              label=f"delta ({len(deltas)} iterations)")
    axes.axhline(theta, color="#d62728", linestyle="--", linewidth=1.2,
                 label=f"theta = {theta:g}")

    axes.set_yscale("log")
    axes.set_title(title)
    axes.set_xlabel("Iteration")
    axes.set_ylabel("delta = max |V_moi - V_cu|  (thang log)")
    axes.grid(True, which="both", linestyle=":", alpha=0.7)
    axes.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def run_one_setting(map_name: str, is_slippery: bool, gamma: float,
                    theta: float, max_iterations: int, n_episodes: int,
                    seed: int) -> dict:
    """Giai mot cau hinh FrozenLake bang ca hai thuat toan va in bao cao."""
    env = create_environment(map_name=map_name, is_slippery=is_slippery)
    n_states, n_actions = env_sizes(env)

    print("=" * 74)
    print(f"FrozenLake-v1 {map_name}, is_slippery={is_slippery}")
    print(f"gamma={gamma}, theta={theta}, max_iterations={max_iterations}")
    print(f"{n_states} state, {n_actions} action")
    print("=" * 74)
    print()

    # --- Baseline: uniform random policy duoc danh gia bang Policy Evaluation
    random_policy = uniform_random_policy(n_states, n_actions)
    V_random, n_eval = policy_evaluation(env, random_policy, gamma, theta,
                                         max_iterations)
    print(f"Policy Evaluation cho uniform random policy: {n_eval} iteration, "
          f"V(0) = {V_random[0]:.6f}")
    print()

    # --- Value Iteration
    start = perf_counter()
    V_vi, n_vi, deltas = value_iteration(env, gamma, theta, max_iterations)
    policy_vi = greedy_policy_from_value(env, V_vi, gamma)
    time_vi = perf_counter() - start

    # --- Policy Iteration
    start = perf_counter()
    policy_pi, V_pi, n_pi, history = policy_iteration(
        env, gamma, theta, max_iterations, track_history=True)
    time_pi = perf_counter() - start

    print("Optimal state values (Value Iteration):")
    print_value_grid(env, V_vi)
    print()
    print("Optimal policy:")
    print_policy(env, policy_vi)
    print()

    evaluation_vi = evaluate_policy_by_simulation(env, policy_vi, n_episodes, seed)
    evaluation_pi = evaluate_policy_by_simulation(env, policy_pi, n_episodes, seed)

    print("| Thuat toan       | Vong lap | Bellman sweep | Thoi gian (ms) | Success rate | Mean reward |")
    print("|------------------|---------:|--------------:|---------------:|-------------:|------------:|")
    print(f"| Value Iteration  | {n_vi:>9d}| {n_vi:>14d}| {time_vi * 1000:>15.2f}| "
          f"{evaluation_vi['success_rate']:>13.4f}| {evaluation_vi['mean_reward']:>12.4f}|")
    total_sweeps = int(sum(history["eval_iterations"]))
    print(f"| Policy Iteration | {n_pi:>9d}| {total_sweeps:>14d}| {time_pi * 1000:>15.2f}| "
          f"{evaluation_pi['success_rate']:>13.4f}| {evaluation_pi['mean_reward']:>12.4f}|")
    print()
    print(f"Hai policy giong nhau: {bool(np.array_equal(policy_vi, policy_pi))}")
    print(f"Sai lech V lon nhat  : {float(np.max(np.abs(V_vi - V_pi))):.2e}")

    suffix = f"{map_name}_{'slippery' if is_slippery else 'deterministic'}"
    figure_path = plot_convergence(
        deltas, theta,
        f"Value Iteration tren FrozenLake-v1 {map_name} (is_slippery={is_slippery})",
        FIGURES_DIR / f"main_convergence_{suffix}.png")
    print(f"Figure saved to      : {figure_path}")
    print()

    env.close()

    return {
        "map_name": map_name,
        "is_slippery": is_slippery,
        "n_vi": n_vi,
        "n_pi": n_pi,
        "total_sweeps": total_sweeps,
        "time_vi": time_vi,
        "time_pi": time_pi,
        "success_vi": evaluation_vi["success_rate"],
        "success_pi": evaluation_pi["success_rate"],
    }


def parse_arguments() -> argparse.Namespace:
    """Doc tham so dong lenh: gamma, theta, max_iterations, ban do, che do tron."""
    parser = argparse.ArgumentParser(
        description="Dynamic Programming solver cho FrozenLake-v1 (Lab02)")
    parser.add_argument("--map-name", default="4x4", choices=["4x4", "8x8"],
                        help="kich thuoc ban do FrozenLake")
    parser.add_argument("--no-slippery", action="store_true",
                        help="dung is_slippery=False (moi truong tat dinh)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor")
    parser.add_argument("--theta", type=float, default=1e-8,
                        help="nguong dung cua cac thuat toan DP")
    parser.add_argument("--max-iterations", type=int, default=10000,
                        help="so vong lap toi da")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="so episode dung de danh gia policy")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--all", action="store_true",
                        help="chay ca hai cau hinh is_slippery=False va True")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    settings = ([False, True] if arguments.all
                else [not arguments.no_slippery])

    summaries = []
    for is_slippery in settings:
        summaries.append(run_one_setting(
            map_name=arguments.map_name,
            is_slippery=is_slippery,
            gamma=arguments.gamma,
            theta=arguments.theta,
            max_iterations=arguments.max_iterations,
            n_episodes=arguments.episodes,
            seed=arguments.seed,
        ))

    if len(summaries) > 1:
        print("=" * 74)
        print("TONG HOP")
        print("=" * 74)
        print("| Ban do | slippery | VI vong | PI vong | PI sweep | VI ms  | PI ms  | Success |")
        print("|--------|----------|--------:|--------:|---------:|-------:|-------:|--------:|")
        for summary in summaries:
            print(f"| {summary['map_name']:<7s}| "
                  f"{str(summary['is_slippery']):<9s}| {summary['n_vi']:>8d}| "
                  f"{summary['n_pi']:>8d}| {summary['total_sweeps']:>9d}| "
                  f"{summary['time_vi'] * 1000:>7.2f}| {summary['time_pi'] * 1000:>7.2f}| "
                  f"{summary['success_vi']:>8.3f}|")
        print()

    print("Ghi chu: chuong trinh mini-project day du cua Bai 36 nam o "
          "src/bai36.py (chay: python src/bai36.py).")


if __name__ == "__main__":
    main()
