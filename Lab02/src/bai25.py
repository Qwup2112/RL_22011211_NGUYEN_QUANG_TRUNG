"""Bai 25 - Theo doi hoi tu cua Policy Evaluation.

Moi iteration luu lai delta = max|new_V - V| va ve do thi.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mdp_utils import (create_environment, env_sizes, policy_evaluation,  # noqa: E402
                       uniform_random_policy)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def plot_evaluation_convergence(deltas_by_gamma: dict, theta: float,
                                output_path: Path) -> None:
    """Ve delta theo iteration (truc y logarithm) cho nhieu gamma."""
    figure, axes = plt.subplots(figsize=(10, 5.5))

    for gamma, deltas in deltas_by_gamma.items():
        axes.plot(range(1, len(deltas) + 1), deltas, linewidth=1.8,
                  label=f"gamma = {gamma} ({len(deltas)} iterations)")

    axes.axhline(theta, color="#d62728", linestyle="--", linewidth=1.2,
                 label=f"theta = {theta:g}")

    axes.set_yscale("log")
    axes.set_title("Policy Evaluation: hoi tu cua delta theo iteration")
    axes.set_xlabel("Iteration")
    axes.set_ylabel("delta = max |V_moi - V_cu|  (thang log)")
    axes.grid(True, which="both", linestyle=":", alpha=0.7)
    axes.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    n_states, n_actions = env_sizes(env)

    policy = uniform_random_policy(n_states, n_actions)
    theta = 1e-8

    deltas_by_gamma = {}
    print("gamma | iterations | delta cuoi cung | V(0)")
    print("------+------------+-----------------+-----------")
    for gamma in [0.90, 0.95, 0.99]:
        V, n_iterations, deltas = policy_evaluation(
            env, policy, gamma=gamma, theta=theta, track_deltas=True)
        deltas_by_gamma[gamma] = deltas
        print(f"{gamma:^6.2f}| {n_iterations:^11d}| {deltas[-1]:^16.2e}| {V[0]:.8f}")

    print()
    print("10 delta dau tien voi gamma = 0.99:")
    print(np.round(deltas_by_gamma[0.99][:10], 8))

    output_path = FIGURES_DIR / "policy_evaluation_convergence.png"
    plot_evaluation_convergence(deltas_by_gamma, theta, output_path)
    print()
    print(f"Figure saved to: {output_path}")

    print()
    print("Nhan xet: delta giam theo cap so nhan (duong thang tren truc log),")
    print("he so giam xap xi gamma. gamma cang gan 1 thi hoi tu cang cham vi")
    print("thong tin phai lan truyen xa hon truoc khi value on dinh.")

    env.close()


if __name__ == "__main__":
    main()
