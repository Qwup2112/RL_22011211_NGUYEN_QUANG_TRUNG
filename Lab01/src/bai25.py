"""Bai 25 - Anh xa action cua FrozenLake sang ten co nghia.

Y nghia cua 0/1/2/3 duoc xac dinh bang thi nghiem: dat agent vao mot o giua
ban do, thuc hien tung action va quan sat o moi (voi is_slippery=False nen
ket qua la tat dinh).
"""

import gymnasium as gym

# Ket qua thi nghiem ben duoi xac nhan bang anh xa chuan cua FrozenLake
ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP",
}


def discover_action_meaning(env: gym.Env, start_state: int = 9) -> dict[int, str]:
    """Thu tung action tu mot o giua ban do va suy ra huong di chuyen."""
    n_cols = env.unwrapped.desc.shape[1]
    discovered = {}

    for action in range(int(env.action_space.n)):
        env.reset()
        # Dat agent vao o start_state (o giua, khong sat bien) de moi huong
        # deu tao ra thay doi quan sat duoc
        env.unwrapped.s = start_state

        next_state, reward, terminated, truncated, info = env.step(action)

        old_row, old_col = divmod(start_state, n_cols)
        new_row, new_col = divmod(int(next_state), n_cols)
        delta = (new_row - old_row, new_col - old_col)

        direction = {
            (0, -1): "LEFT",
            (1, 0): "DOWN",
            (0, 1): "RIGHT",
            (-1, 0): "UP",
            (0, 0): "NO MOVE (blocked by the wall)",
        }[delta]

        discovered[action] = direction

    return discovered


def main() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env.reset(seed=42)
    env.action_space.seed(42)

    discovered = discover_action_meaning(env, start_state=9)

    print("Experiment: start from state 9 (row 2, col 1) and try every action")
    print()
    print("Action | Observed direction | ACTION_NAMES")
    print("-------+--------------------+-------------")
    for action, direction in discovered.items():
        print(f"{action:^7d}| {direction:<19s}| {ACTION_NAMES[action]}")

    print()
    print("ACTION_NAMES =", ACTION_NAMES)

    # Sinh mot action ngau nhien va in ten cua no
    print()
    for _ in range(5):
        action = int(env.action_space.sample())
        print(f"Action {action} -> {ACTION_NAMES[action]}")

    env.close()


if __name__ == "__main__":
    main()
