"""Bai 29 - Policy Iteration hoan chinh.

    Khoi tao policy -> Policy Evaluation -> Policy Improvement
                    -> policy on dinh? -> lap hoac ket thuc

Ham policy_iteration() nam trong src/mdp_utils.py, tra ve
(policy, V, n_policy_iterations).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mdp_utils import (create_environment, policy_iteration,  # noqa: E402
                       print_policy_grid, print_value_grid)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def plot_policy_iteration_convergence(history: dict, output_path: Path) -> None:
    """Ve gia tri trung binh va so state doi action theo tung vong Policy Iteration."""
    iterations = np.arange(1, len(history["mean_value"]) + 1)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

    axes[0].plot(iterations, history["mean_value"], marker="o", linewidth=2.0,
                 color="#1f77b4", label="mean V(s)")
    axes[0].set_title("Policy Iteration: gia tri trung binh cua V theo tung vong")
    axes[0].set_ylabel("mean V(s)")
    axes[0].grid(True, linestyle=":", alpha=0.7)
    axes[0].legend()

    axes[1].bar(iterations, history["n_changed"], color="#d62728", alpha=0.85,
                label="so state doi action")
    axes[1].plot(iterations, history["eval_iterations"], marker="s",
                 color="#2ca02c", label="so iteration cua Policy Evaluation")
    axes[1].set_title("Policy Improvement: so state doi action va chi phi danh gia")
    axes[1].set_xlabel("Vong Policy Iteration")
    axes[1].set_ylabel("So luong")
    axes[1].grid(True, linestyle=":", alpha=0.7)
    axes[1].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    gamma = 0.99
    theta = 1e-8

    policy, V, n_policy_iterations, history = policy_iteration(
        env, gamma=gamma, theta=theta, track_history=True)

    print(f"Policy Iteration tren FrozenLake-v1 4x4 (is_slippery=True)")
    print(f"gamma = {gamma}, theta = {theta}")
    print(f"So vong Policy Iteration: {n_policy_iterations}")
    print(f"Tong so sweep cua Policy Evaluation: {sum(history['eval_iterations'])}")
    print()
    print("Optimal state values:")
    print_value_grid(env, V)
    print()
    print("Optimal policy:")
    print_policy_grid(env, policy)
    print()
    print("policy (vector):", policy)
    print()

    print("Vong | mean V(s) | state doi action | eval iterations")
    print("-----+-----------+------------------+----------------")
    for index in range(n_policy_iterations):
        print(f"{index + 1:^5d}| {history['mean_value'][index]:^10.6f}| "
              f"{history['n_changed'][index]:^17d}| "
              f"{history['eval_iterations'][index]:^15d}")

    output_path = FIGURES_DIR / "policy_iteration_convergence.png"
    plot_policy_iteration_convergence(history, output_path)
    print()
    print(f"Figure saved to: {output_path}")

    env.close()


if __name__ == "__main__":
    main()
