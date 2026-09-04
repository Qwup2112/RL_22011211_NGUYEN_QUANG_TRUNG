"""Bai 23 - Tao FrozenLake-v1 (is_slippery=False) va kham pha cac khong gian."""

import gymnasium as gym
from gymnasium import spaces


def count_discrete(space: spaces.Space) -> int:
    """Dem so phan tu cua mot khong gian roi rac (Discrete)."""
    if not isinstance(space, spaces.Discrete):
        raise TypeError(f"Khong phai khong gian roi rac: {space}")
    return int(space.n)


def main() -> None:
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False,
    )

    print("Observation space:", env.observation_space)
    print("Action space     :", env.action_space)
    print()

    n_states = count_discrete(env.observation_space)
    n_actions = count_discrete(env.action_space)

    print(f"Number of states : {n_states}")
    print(f"Number of actions: {n_actions}")

    # Ban do mac dinh la luoi 4x4 -> 16 o = 16 state
    desc = env.unwrapped.desc
    print(f"Map shape        : {desc.shape}")
    print()
    print("Map layout:")
    for row_index, row in enumerate(desc):
        cells = " ".join(cell.decode("ascii") for cell in row)
        print(f"  row {row_index}: {cells}")

    print()
    print("State index = row * n_cols + col")

    env.close()


if __name__ == "__main__":
    main()
