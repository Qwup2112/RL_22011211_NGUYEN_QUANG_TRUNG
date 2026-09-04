"""Bai 32 - Value Iteration hoan chinh.

Lap Bellman optimality backup den khi delta < theta.
Ham value_iteration() nam trong src/mdp_utils.py, tra ve (V, n_iterations, deltas).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mdp_utils import (create_environment, print_value_grid,  # noqa: E402
                       value_iteration)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def plot_value_iteration_convergence(deltas_by_gamma: dict, theta: float,
                                     output_path: Path) -> None:
    """Ve delta cua Value Iteration theo iteration (truc y log)."""
    figure, axes = plt.subplots(figsize=(10, 5.5))

    for gamma, deltas in deltas_by_gamma.items():
        axes.plot(range(1, len(deltas) + 1), deltas, linewidth=1.8,
                  label=f"gamma = {gamma} ({len(deltas)} iterations)")

    axes.axhline(theta, color="#d62728", linestyle="--", linewidth=1.2,
                 label=f"theta = {theta:g}")

    axes.set_yscale("log")
    axes.set_title("Value Iteration: hoi tu cua delta theo iteration")
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
    theta = 1e-8

    V, n_iterations, deltas = value_iteration(env, gamma=0.99, theta=theta)

    print("Value Iteration tren FrozenLake-v1 4x4 (is_slippery=True)")
    print(f"gamma = 0.99, theta = {theta}")
    print(f"Hoi tu sau {n_iterations} iteration, delta cuoi = {deltas[-1]:.2e}")
    print()
    print("Optimal state values V*:")
    print_value_grid(env, V)
    print()
    print("V* (vector):", np.round(V, 6))
    print()

    deltas_by_gamma = {}
    print("gamma | iterations | V*(0)     | delta cuoi")
    print("------+------------+-----------+-----------")
    for gamma in [0.90, 0.95, 0.99]:
        V_g, n_g, deltas_g = value_iteration(env, gamma=gamma, theta=theta)
        deltas_by_gamma[gamma] = deltas_g
        print(f"{gamma:^6.2f}| {n_g:^11d}| {V_g[0]:^10.6f}| {deltas_g[-1]:.2e}")

    output_path = FIGURES_DIR / "value_iteration_convergence.png"
    plot_value_iteration_convergence(deltas_by_gamma, theta, output_path)
    print()
    print(f"Figure saved to: {output_path}")

    n_low = len(deltas_by_gamma[0.90])
    n_high = len(deltas_by_gamma[0.99])
    print()
    print("Nhan xet: delta giam gan nhu cap so nhan voi he so gamma, nen gamma cang")
    print(f"gan 1 thi so iteration can thiet cang lon: {n_low} iteration voi gamma=0.90")
    print(f"nhung {n_high} iteration voi gamma=0.99 (gap {n_high / n_low:.1f} lan).")

    env.close()


if __name__ == "__main__":
    main()
