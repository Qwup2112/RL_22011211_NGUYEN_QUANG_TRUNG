"""Bai 36 - Mini-project: Dynamic Programming Solver cho FrozenLake-v1.

Chuong trinh hoan chinh gom day du cac ham theo yeu cau cua de bai:
    create_environment()             - tao moi truong
    get_transition_model()           - lay P[state][action]
    q_from_v()                       - mot Bellman backup
    policy_evaluation()              - Iterative Policy Evaluation
    greedy_policy_from_value()       - Policy Improvement
    policy_iteration()               - Policy Iteration
    value_iteration()                - Value Iteration
    evaluate_policy_by_simulation()  - danh gia bang mo phong
    print_policy()                   - in policy dang luoi
    plot_convergence()               - ve duong hoi tu
    main()                           - dieu phoi toan bo thi nghiem

Cac thuat toan duoc cai dat trong src/mdp_utils.py (muc 14 cua de bai) va
import lai o day - khong dung bat ky thu vien RL nao co san Value Iteration
hay Policy Iteration.

Chuong trinh chay ca hai cau hinh is_slippery=False va is_slippery=True,
co tham so gamma, theta, max_iterations.
"""

from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Import lai cac ham dung chung thay vi copy-paste code
from mdp_utils import (create_environment, env_sizes,  # noqa: E402
                       evaluate_policy_by_simulation, get_transition_model,
                       greedy_policy_from_value, policy_evaluation,
                       policy_iteration, print_policy, print_policy_grid,
                       print_value_grid, q_from_v, value_iteration)

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"

GAMMA = 0.99
THETA = 1e-8
MAX_ITERATIONS = 10000
N_EVAL_EPISODES = 1000
SEED = 42


def solve_with_value_iteration(env, gamma: float, theta: float,
                               max_iterations: int) -> dict:
    """Giai MDP bang Value Iteration va do thoi gian chay."""
    start = perf_counter()
    V, n_iterations, deltas = value_iteration(env, gamma=gamma, theta=theta,
                                              max_iterations=max_iterations)
    policy = greedy_policy_from_value(env, V, gamma)
    elapsed = perf_counter() - start

    return {
        "name": "Value Iteration",
        "V": V,
        "policy": policy,
        "n_iterations": n_iterations,
        "total_sweeps": n_iterations,
        "deltas": deltas,
        "time": elapsed,
    }


def solve_with_policy_iteration(env, gamma: float, theta: float,
                                max_iterations: int) -> dict:
    """Giai MDP bang Policy Iteration va do thoi gian chay."""
    start = perf_counter()
    policy, V, n_iterations, history = policy_iteration(
        env, gamma=gamma, theta=theta,
        max_iterations=max_iterations, track_history=True)
    elapsed = perf_counter() - start

    return {
        "name": "Policy Iteration",
        "V": V,
        "policy": policy,
        "n_iterations": n_iterations,
        "total_sweeps": int(sum(history["eval_iterations"])),
        "history": history,
        "time": elapsed,
    }


def plot_convergence(results_by_setting: dict, theta: float,
                     output_path: Path) -> Path:
    """Ve duong hoi tu cua Value Iteration cho tung cau hinh moi truong."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    for index, (setting, results) in enumerate(results_by_setting.items()):
        deltas = results["Value Iteration"]["deltas"]
        axes[0].plot(range(1, len(deltas) + 1), deltas, linewidth=1.8,
                     label=f"{setting} ({len(deltas)} iterations)")

        history = results["Policy Iteration"]["history"]
        axes[1].plot(range(1, len(history["mean_value"]) + 1),
                     history["mean_value"], marker="o", linewidth=1.8,
                     label=f"{setting} ({len(history['mean_value'])} vong)")

    axes[0].axhline(theta, color="#d62728", linestyle="--", linewidth=1.2,
                    label=f"theta = {theta:g}")
    axes[0].set_yscale("log")
    axes[0].set_title("Value Iteration: delta theo iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("delta = max |V_moi - V_cu| (log)")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.7)
    axes[0].legend()

    axes[1].set_title("Policy Iteration: mean V(s) theo tung vong")
    axes[1].set_xlabel("Vong Policy Iteration")
    axes[1].set_ylabel("mean V(s)")
    axes[1].grid(True, linestyle=":", alpha=0.7)
    axes[1].legend()

    figure.suptitle("Mini-project: hoi tu cua Dynamic Programming tren FrozenLake-v1")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_convergence_data(results_by_setting: dict) -> Path:
    """Luu chuoi delta cua Value Iteration ra file CSV trong data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "value_iteration_deltas.csv"

    with output_path.open("w", encoding="utf-8") as file:
        file.write("setting,iteration,delta\n")
        for setting, results in results_by_setting.items():
            for iteration, delta in enumerate(results["Value Iteration"]["deltas"], start=1):
                file.write(f"{setting},{iteration},{delta}\n")

    return output_path


def report_solution(env, result: dict, evaluation: dict) -> None:
    """In value table, policy va ket qua danh gia cua mot thuat toan."""
    print(f"--- {result['name']} ---")
    print(f"  So vong lap chinh : {result['n_iterations']}")
    print(f"  Tong Bellman sweep: {result['total_sweeps']}")
    print(f"  Thoi gian         : {result['time'] * 1000:.2f} ms")
    print(f"  V*(state 0)       : {result['V'][0]:.6f}")
    print("  Value table:")
    print_value_grid(env, result["V"])
    print("  Optimal policy:")
    print_policy_grid(env, result["policy"])
    print(f"  Success rate      : {evaluation['success_rate']:.4f} "
          f"({evaluation['success']}/{evaluation['n_episodes']} episode)")
    print(f"  Mean reward       : {evaluation['mean_reward']:.4f}")
    print(f"  Episode length    : mean {evaluation['mean_length']:.2f}, "
          f"min {evaluation['min_length']}, max {evaluation['max_length']}")
    print()


def solve_setting(is_slippery: bool, gamma: float, theta: float,
                  max_iterations: int, n_episodes: int, seed: int) -> dict:
    """Giai FrozenLake bang ca hai thuat toan cho mot cau hinh moi truong."""
    env = create_environment(map_name="4x4", is_slippery=is_slippery)
    n_states, n_actions = env_sizes(env)

    print("=" * 72)
    print(f"FrozenLake-v1 4x4, is_slippery={is_slippery}, "
          f"gamma={gamma}, theta={theta}")
    print(f"{n_states} state, {n_actions} action")
    print("=" * 72)

    # Kiem tra da doc duoc model chuyen trang thai
    transition_model = get_transition_model(env)
    n_transitions = sum(len(transition_model[s][a])
                        for s in range(n_states) for a in range(n_actions))
    print(f"Model co {n_transitions} transition tren {n_states * n_actions} "
          f"cap (state, action)")
    print()

    results = {}
    evaluations = {}

    for solver in (solve_with_value_iteration, solve_with_policy_iteration):
        result = solver(env, gamma, theta, max_iterations)
        evaluation = evaluate_policy_by_simulation(
            env, result["policy"], n_episodes=n_episodes, seed=seed)

        results[result["name"]] = result
        evaluations[result["name"]] = evaluation
        report_solution(env, result, evaluation)

    vi = results["Value Iteration"]
    pi = results["Policy Iteration"]
    print("So sanh hai thuat toan:")
    print(f"  Policy giong nhau : {bool(np.array_equal(vi['policy'], pi['policy']))}")
    print(f"  Sai lech V lon nhat: {float(np.max(np.abs(vi['V'] - pi['V']))):.2e}")
    print(f"  Thoi gian VI / PI  : {vi['time'] * 1000:.2f} ms / "
          f"{pi['time'] * 1000:.2f} ms")
    print()

    env.close()

    results["evaluations"] = evaluations
    return results


def main() -> None:
    settings = {
        "is_slippery=False": False,
        "is_slippery=True": True,
    }

    results_by_setting = {}
    for label, is_slippery in settings.items():
        results_by_setting[label] = solve_setting(
            is_slippery=is_slippery,
            gamma=GAMMA,
            theta=THETA,
            max_iterations=MAX_ITERATIONS,
            n_episodes=N_EVAL_EPISODES,
            seed=SEED,
        )

    # Luu du lieu hoi tu va ve bieu do
    data_path = save_convergence_data(results_by_setting)
    figure_path = plot_convergence(results_by_setting, THETA,
                                   FIGURES_DIR / "mini_project_convergence.png")

    print("=" * 72)
    print("BANG TONG HOP")
    print("=" * 72)
    print("| Cau hinh          | Thuat toan       | Vong lap | Sweep | ms     | Success |")
    print("|-------------------|------------------|---------:|------:|-------:|--------:|")
    for label, results in results_by_setting.items():
        for name in ["Value Iteration", "Policy Iteration"]:
            result = results[name]
            evaluation = results["evaluations"][name]
            print(f"| {label:<18s}| {name:<17s}| {result['n_iterations']:>9d}| "
                  f"{result['total_sweeps']:>6d}| {result['time'] * 1000:>7.2f}| "
                  f"{evaluation['success_rate']:>8.3f}|")

    print()
    print(f"Convergence data saved to: {data_path}")
    print(f"Figure saved to          : {figure_path}")

    # ------------------------------------------------------------------
    # KET LUAN
    # 1. Voi is_slippery=False, DP tim ra duong di ngan nhat va success rate
    #    dat 100%: moi truong tat dinh nen policy toi uu la mot duong di co dinh.
    # 2. Voi is_slippery=True, success rate toi uu chi con khoang 0.72-0.74 -
    #    day KHONG phai loi cua thuat toan ma la gioi han cua chinh bai toan.
    # 3. Policy toi uu o che do tron khong di duong ngan nhat ma di sat tuong
    #    de khi bi truot van khong roi vao ho.
    # 4. Hai thuat toan luon cho cung mot policy va cung V*, khac nhau o chi phi.
    # 5. Dieu kien de dung duoc DP: phai biet TRANSITION MODEL (env.unwrapped.P).
    #    Neu khong biet model thi phai chuyen sang Monte Carlo hoac TD Learning.
    # ------------------------------------------------------------------
    slippery = results_by_setting["is_slippery=True"]["evaluations"]["Value Iteration"]
    deterministic = results_by_setting["is_slippery=False"]["evaluations"]["Value Iteration"]

    print()
    print("KET LUAN")
    print("-" * 72)
    print(f"1. is_slippery=False: success rate {deterministic['success_rate']:.3f} - "
          f"moi truong tat dinh nen DP tim duoc duong di chac chan toi Goal.")
    print(f"2. is_slippery=True : success rate {slippery['success_rate']:.3f} - day la")
    print("   gioi han cua chinh bai toan, khong phai loi thuat toan.")
    print("3. Policy toi uu o che do tron di sat tuong de khi truot khong roi vao ho.")
    print("4. Value Iteration va Policy Iteration luon cho cung policy va cung V*.")
    print("5. DP la model-based: bat buoc phai biet transition model. Khong biet")
    print("   model thi phai dung Monte Carlo hoac Temporal-Difference Learning.")


if __name__ == "__main__":
    main()
