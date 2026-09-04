"""Bai 35 - So sanh ba agent tren CartPole-v1 (500 episode moi agent).

Agent 1: Random policy
Agent 2: Angle-based policy (Bai 31)
Agent 3: Improved policy (Bai 32)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bai31 import angle_based_policy  # noqa: E402
from bai32 import improved_policy  # noqa: E402
from bai33 import make_random_policy  # noqa: E402
from bai34 import evaluate_policy  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def plot_agent_comparison(results: dict, output_path: Path) -> None:
    """Ve bieu do cot so sanh reward trung binh cua cac agent."""
    names = list(results.keys())
    means = [results[name]["mean_reward"] for name in names]
    stds = [results[name]["std_reward"] for name in names]

    figure, axes = plt.subplots(figsize=(8, 5.5))
    bars = axes.bar(names, means, yerr=stds, capsize=8,
                    color=["#8c8c8c", "#1f77b4", "#2ca02c"], alpha=0.9)

    # Dat nhan gia tri phia tren dau mut cua thanh error bar de khong bi de chu
    offset = 0.03 * max(mean + std for mean, std in zip(means, stds))
    for bar, mean, std in zip(bars, means, stds):
        axes.text(bar.get_x() + bar.get_width() / 2, mean + std + offset,
                  f"{mean:.1f}", ha="center", va="bottom", fontweight="bold")

    axes.set_ylim(0, max(mean + std for mean, std in zip(means, stds)) * 1.15)

    axes.set_title("CartPole-v1: mean reward of three agents (500 episodes each)")
    axes.set_xlabel("Agent")
    axes.set_ylabel("Mean total reward (error bar = std)")
    axes.grid(True, axis="y", linestyle=":", alpha=0.7)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    env_name = "CartPole-v1"
    n_episodes = 500
    seed = 42

    policies = {
        "Random": make_random_policy(env_name, seed=seed),
        "Angle-based": angle_based_policy,
        "Improved": improved_policy,
    }

    results = {}
    for name, policy in policies.items():
        results[name] = evaluate_policy(env_name, policy,
                                        n_episodes=n_episodes, seed=seed)

    print(f"{env_name}: {n_episodes} episodes per agent, seed = {seed}")
    print()
    print("Agent       | Mean reward |  Std   |  Min   |  Max   | Mean length")
    print("------------+-------------+--------+--------+--------+------------")
    for name, result in results.items():
        print(f"{name:<12s}| {result['mean_reward']:^12.2f}| "
              f"{result['std_reward']:^7.2f}| {result['min_reward']:^7.1f}| "
              f"{result['max_reward']:^7.1f}| {result['mean_length']:^11.2f}")

    output_path = FIGURES_DIR / "comparison_agents.png"
    plot_agent_comparison(results, output_path)
    print()
    print(f"Figure saved to: {output_path}")

    # ------------------------------------------------------------------
    # NHAN XET (5-10 dong)
    # 1. Random policy chi tru duoc khoang 20-25 buoc: chon action ngau nhien
    #    khong he quan tam den trang thai nen cot nga rat nhanh.
    # 2. Angle-based policy tot hon ro ret vi da dung thong tin observation:
    #    day xe ve phia cot dang nghieng giup keo diem tua ve duoi trong tam.
    # 3. Improved policy (goc + van toc goc) tot nhat vi no phan ung SOM,
    #    du doan goc sap toi thay vi doi den khi goc da lech.
    # 4. Std giam dan tu Random den Improved: Improved co std = 0 vi episode nao
    #    cung cham tran 500 buoc, tuc la no on dinh voi MOI trang thai khoi tao.
    # 5. Random luon ket thuc bang terminated (cot nga) con Improved luon ket thuc
    #    bang truncated (het 500 buoc) - day la ket qua tot nhat cua CartPole-v1.
    # 6. Ca ba agent deu KHONG hoc: khong co tham so nao duoc cap nhat tu reward.
    #    Chung chi la baseline de so sanh voi cac thuat toan RL o cac bai sau.
    # 7. Ket luan: chi can dung them thong tin tu observation la hieu qua da tang
    #    dang ke, day chinh la dieu ma mot policy hoc duoc se lam mot cach tu dong.
    # ------------------------------------------------------------------
    print()
    print("Remarks:")
    print(f"1. Random policy survives only {results['Random']['mean_reward']:.1f} steps "
          f"on average: it completely ignores the state.")
    print(f"2. Angle-based policy reaches {results['Angle-based']['mean_reward']:.1f}: "
          f"using observation[2] alone already doubles the reward.")
    print(f"3. Improved policy reaches {results['Improved']['mean_reward']:.1f}: adding "
          f"observation[3] lets it react before the angle grows.")
    print(f"4. Std drops from {results['Random']['std_reward']:.1f} to "
          f"{results['Improved']['std_reward']:.1f}: the improved rule is stable for")
    print("   every random initial state, not lucky on a few episodes.")
    print(f"5. Random ends with terminated {results['Random']['terminated_count']}/"
          f"{n_episodes} times (the pole falls), Improved ends with truncated "
          f"{results['Improved']['truncated_count']}/{n_episodes} times")
    print("   (the 500-step TimeLimit), which is the best outcome on CartPole-v1.")
    print("6. None of the three agents learns; no parameter is updated from reward.")
    print("7. Using more information from the observation already improves the agent.")


if __name__ == "__main__":
    main()
