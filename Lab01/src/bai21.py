"""Bai 21 - Seed cho action space.

env.action_space.seed(n) dat lai bo sinh so ngau nhien rieng cua action space,
nho do chuoi action sinh ra tai lap duoc giua cac lan chay.
"""

import gymnasium as gym


def sample_action_sequence(seed: int, n_actions: int = 20) -> list[int]:
    """Tao env moi, seed action_space va sinh n_actions action."""
    env = gym.make("CartPole-v1")

    env.action_space.seed(seed)          # cach seed rieng cho action space
    actions = [int(env.action_space.sample()) for _ in range(n_actions)]

    env.close()
    return actions


def main() -> None:
    seed = 123

    # "Chay chuong trinh hai lan": mo phong bang hai lan goi doc lap
    first_run = sample_action_sequence(seed)
    second_run = sample_action_sequence(seed)

    print(f"Seed used: {seed}")
    print("Run 1:", first_run)
    print("Run 2:", second_run)
    print()
    print("Two sequences are identical:", first_run == second_run)

    # Doi chung: khong seed (hoac seed khac) thi chuoi action khac di
    other_run = sample_action_sequence(seed + 1)
    print()
    print(f"Run with seed={seed + 1}:", other_run)
    print("Identical to run 1:", other_run == first_run)


if __name__ == "__main__":
    main()
