"""Bai 4 - Kham pha observation space cua CartPole.

Xac dinh bang code: shape, kieu du lieu, gioi han duoi, gioi han tren.
"""

import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1")

    obs_space = env.observation_space

    print("Observation space :", obs_space)
    print("Type              :", type(obs_space).__name__)
    print("Shape             :", obs_space.shape)
    print("Data type (dtype) :", obs_space.dtype)
    print("Lower bound (low) :", obs_space.low)
    print("Upper bound (high):", obs_space.high)

    # In tung chieu kem y nghia vat ly cua CartPole
    names = [
        "cart position",
        "cart velocity",
        "pole angle (rad)",
        "pole angular velocity",
    ]
    print()
    print("Chi tiet tung chieu:")
    for index, name in enumerate(names):
        print(
            f"  obs[{index}] = {name:<22} "
            f"low={obs_space.low[index]:>12.5f} "
            f"high={obs_space.high[index]:>12.5f}"
        )

    # -inf / +inf nghia la Gymnasium khong rang buoc gia tri cua chieu do
    print()
    print("Sample observation:", obs_space.sample())

    env.close()


if __name__ == "__main__":
    main()
