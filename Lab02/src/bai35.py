"""Bai 35 - So sanh Value Iteration va Policy Iteration.

Do so vong lap, thoi gian chay, success rate va mean reward tren
FrozenLake-v1 4x4 (is_slippery=True).
"""

from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mdp_utils import (create_environment, evaluate_policy_by_simulation,  # noqa: E402
                       greedy_policy_from_value, policy_iteration,
                       print_policy_grid, value_iteration)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def run_value_iteration(env, gamma: float, theta: float) -> dict:
    """Chay Value Iteration va do thoi gian."""
    start = perf_counter()
    V, n_iterations, deltas = value_iteration(env, gamma=gamma, theta=theta)
    policy = greedy_policy_from_value(env, V, gamma)
    elapsed = perf_counter() - start

    return {"name": "Value Iteration", "V": V, "policy": policy,
            "n_iterations": n_iterations, "time": elapsed, "deltas": deltas}


def run_policy_iteration(env, gamma: float, theta: float) -> dict:
    """Chay Policy Iteration va do thoi gian."""
    start = perf_counter()
    policy, V, n_iterations, history = policy_iteration(
        env, gamma=gamma, theta=theta, track_history=True)
    elapsed = perf_counter() - start

    return {"name": "Policy Iteration", "V": V, "policy": policy,
            "n_iterations": n_iterations, "time": elapsed,
            "total_sweeps": int(sum(history["eval_iterations"]))}


def plot_algorithm_comparison(results: list, output_path: Path) -> None:
    """Ve bieu do so sanh so vong lap, thoi gian va success rate."""
    names = [result["name"] for result in results]
    iterations = [result["n_iterations"] for result in results]
    sweeps = [result.get("total_sweeps", result["n_iterations"]) for result in results]
    times = [result["time"] * 1000 for result in results]
    success = [result["evaluation"]["success_rate"] for result in results]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = ["#1f77b4", "#ff7f0e"]

    axes[0].bar(names, iterations, color=colors, alpha=0.9, label="Vong lap chinh")
    axes[0].plot(names, sweeps, "o--", color="#d62728", label="Tong Bellman sweep")
    axes[0].set_yscale("log")
    axes[0].set_title("So vong lap (thang log)")
    axes[0].set_ylabel("So luong")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.7)
    axes[0].legend()

    bars = axes[1].bar(names, times, color=colors, alpha=0.9)
    for bar, value in zip(bars, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value,
                     f"{value:.1f} ms", ha="center", va="bottom", fontweight="bold")
    axes[1].set_title("Thoi gian chay")
    axes[1].set_ylabel("Thoi gian (ms)")
    axes[1].set_ylim(0, max(times) * 1.25)
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.7)

    bars = axes[2].bar(names, success, color=colors, alpha=0.9)
    for bar, value in zip(bars, success):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value,
                     f"{value:.3f}", ha="center", va="bottom", fontweight="bold")
    axes[2].set_title("Success rate (1000 episode)")
    axes[2].set_ylabel("Ty le thang")
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(True, axis="y", linestyle=":", alpha=0.7)

    figure.suptitle("FrozenLake-v1 4x4 (is_slippery=True): Value Iteration vs Policy Iteration")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    gamma = 0.99
    theta = 1e-8
    n_episodes = 1000

    results = [
        run_value_iteration(env, gamma, theta),
        run_policy_iteration(env, gamma, theta),
    ]

    for result in results:
        result["evaluation"] = evaluate_policy_by_simulation(
            env, result["policy"], n_episodes=n_episodes, seed=42)

    print(f"FrozenLake-v1 4x4, is_slippery=True, gamma={gamma}, theta={theta}")
    print(f"Danh gia bang {n_episodes} episode, seed = 42")
    print()
    print("| Thuat toan       | So vong lap | Bellman sweep | Thoi gian (ms) | Success rate | Mean reward |")
    print("|------------------|------------:|--------------:|---------------:|-------------:|------------:|")
    for result in results:
        sweeps = result.get("total_sweeps", result["n_iterations"])
        print(f"| {result['name']:<17s}| {result['n_iterations']:>12d}| {sweeps:>14d}| "
              f"{result['time'] * 1000:>15.2f}| "
              f"{result['evaluation']['success_rate']:>13.4f}| "
              f"{result['evaluation']['mean_reward']:>12.4f}|")

    vi, pi = results
    print()
    print("Policy cua Value Iteration:")
    print_policy_grid(env, vi["policy"])
    print()
    print("Policy cua Policy Iteration:")
    print_policy_grid(env, pi["policy"])
    print()
    print("Hai policy giong nhau:", bool(np.array_equal(vi["policy"], pi["policy"])))
    print("Sai lech V lon nhat  :", float(np.max(np.abs(vi["V"] - pi["V"]))))

    output_path = FIGURES_DIR / "algorithm_comparison.png"
    plot_algorithm_comparison(results, output_path)
    print()
    print(f"Figure saved to: {output_path}")

    # ------------------------------------------------------------------
    # NHAN XET (>= 8 dong)
    # 1. Hai thuat toan cho ra CUNG MOT optimal policy va cung mot V* (sai lech
    #    chi o muc sai so cua theta), dung nhu ly thuyet: ca hai deu hoi tu ve
    #    nghiem duy nhat cua Bellman optimality equation.
    # 2. Success rate cua ca hai deu ~0.72-0.74 - day la muc toi da co the dat
    #    tren FrozenLake tron, phan con lai la do moi truong day agent xuong ho.
    # 3. Policy Iteration can rat it vong lap chinh (7 vong) so voi Value
    #    Iteration (438 iteration), vi moi vong no giai gan nhu chinh xac V_pi.
    # 4. Nhung neu dem theo TONG SO Bellman sweep thi Policy Iteration lai ton
    #    hon nhieu: moi vong phai chay Policy Evaluation den hoi tu.
    # 5. Do do tren ban do 4x4 nay Value Iteration chay NHANH HON han (khoang
    #    3 lan): so vong lap chinh it khong co nghia la re hon, chi phi that su
    #    nam o tong so Bellman backup.
    # 6. Policy Iteration co uu the khi khong gian action lon hoac khi ta co
    #    cach giai he phuong trinh tuyen tinh cho Policy Evaluation.
    # 7. Value Iteration don gian hon ve cai dat: chi mot vong lap, khong can
    #    tach hai pha evaluation/improvement.
    # 8. Ca hai deu la model-based: bat buoc phai truy cap env.unwrapped.P.
    #    Neu khong biet transition model thi khong dung truc tiep duoc, phai
    #    chuyen sang Monte Carlo hoac Temporal-Difference Learning.
    # 9. gamma anh huong manh den chi phi: gamma cang gan 1 thi ca hai thuat
    #    toan deu can nhieu sweep hon vi thong tin phai lan truyen xa hon.
    # ------------------------------------------------------------------
    print()
    print("NHAN XET:")
    print("1. Hai thuat toan cho cung mot optimal policy va cung V* -> dung ly thuyet.")
    print(f"2. Success rate ~{vi['evaluation']['success_rate']:.2f} la muc toi da tren ban do tron.")
    print(f"3. Policy Iteration chi can {pi['n_iterations']} vong chinh, "
          f"Value Iteration can {vi['n_iterations']} iteration.")
    print(f"4. Nhung tong Bellman sweep cua Policy Iteration la {pi['total_sweeps']}, "
          f"nhieu hon {vi['n_iterations']} cua Value Iteration.")
    speedup = pi["time"] / vi["time"]
    print(f"5. Vi the Value Iteration chay nhanh hon {speedup:.1f} lan "
          f"({vi['time'] * 1000:.1f} ms so voi {pi['time'] * 1000:.1f} ms):")
    print("   chi phi that su nam o tong so Bellman backup, khong phai so vong chinh.")
    print("6. Policy Iteration loi the khi so action lon hoac giai duoc he tuyen tinh.")
    print("7. Value Iteration don gian hon: chi mot vong lap duy nhat.")
    print("8. Ca hai deu model-based, bat buoc phai co env.unwrapped.P.")
    print("9. gamma cang gan 1 thi ca hai deu can nhieu sweep hon.")

    env.close()


if __name__ == "__main__":
    main()
