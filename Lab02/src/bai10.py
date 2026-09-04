"""Bai 10 - Anh huong cua discount factor gamma len return."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bai07 import compute_return  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def plot_return_vs_gamma(rewards, gammas, output_path: Path) -> None:
    """Ve G_0 theo gamma cho mot chuoi reward."""
    returns = [compute_return(rewards, gamma) for gamma in gammas]

    figure, axes = plt.subplots(figsize=(10, 5.5))
    axes.plot(gammas, returns, linewidth=2.2, color="#1f77b4",
              label=f"rewards = {list(rewards)}")

    # Danh dau vai moc gamma hay dung
    for gamma in [0.5, 0.9, 0.99]:
        value = compute_return(rewards, gamma)
        axes.plot([gamma], [value], "o", color="#d62728")
        axes.annotate(f"gamma={gamma}\nG_0={value:.3f}", (gamma, value),
                      textcoords="offset points", xytext=(-10, 14), fontsize=9)

    axes.set_title("Anh huong cua discount factor len return G_0")
    axes.set_xlabel("gamma")
    axes.set_ylabel("G_0")
    axes.grid(True, linestyle=":", alpha=0.7)
    axes.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    rewards = [0, 0, 0, 0, 10]
    gammas = np.linspace(0, 1, 101)

    print("rewards =", rewards)
    print("Reward duy nhat nam o buoc cuoi (t = 4) nen G_0 = 10 * gamma^4")
    print()
    print("| Gamma | G_0     | 10*gamma^4 |")
    print("|------:|--------:|-----------:|")
    for gamma in [0.0, 0.25, 0.5, 0.9, 0.99, 1.0]:
        print(f"| {gamma:5.2f} | {compute_return(rewards, gamma):7.4f} | "
              f"{10 * gamma ** 4:10.4f} |")

    output_path = FIGURES_DIR / "gamma_comparison.png"
    plot_return_vs_gamma(rewards, gammas, output_path)
    print()
    print(f"Figure saved to: {output_path}")

    # Nhan xet: duong cong tang rat cham khi gamma nho roi doc len gan gamma = 1.
    # Voi gamma = 0.5 reward xa 4 buoc chi con gia tri 0.625 tren 10, tuc agent
    # gan nhu bo qua no; voi gamma = 0.99 thi con 9.606, gan nhu nguyen ven.


if __name__ == "__main__":
    main()
