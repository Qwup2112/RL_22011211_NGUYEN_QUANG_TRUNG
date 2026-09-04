"""Bai 24 - Hien thi FrozenLake dang text voi render_mode="ansi"."""

import gymnasium as gym


def main() -> None:
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False,
        render_mode="ansi",
    )

    observation, info = env.reset(seed=42)

    print(f"Initial state (observation): {observation}")
    print(f"Info: {info}")
    print()
    print("env.render() output:")
    # Voi render_mode="ansi", env.render() tra ve mot chuoi text
    print(env.render())

    print("Meaning of the symbols:")
    print("  S = Start  : o xuat phat (state 0)")
    print("  F = Frozen : o bang an toan, di qua duoc")
    print("  H = Hole   : ho bang, roi vao la episode terminated (reward 0)")
    print("  G = Goal   : dich den, toi noi thi terminated voi reward 1")
    print()

    # Liet ke vi tri cua tung loai o bang code
    desc = env.unwrapped.desc
    n_rows, n_cols = desc.shape
    positions = {"S": [], "F": [], "H": [], "G": []}
    for row in range(n_rows):
        for col in range(n_cols):
            symbol = desc[row][col].decode("ascii")
            positions[symbol].append(row * n_cols + col)

    for symbol, states in positions.items():
        print(f"  {symbol}: states {states}")

    # Di thu mot buoc DOWN de thay o mau (vi tri agent) di chuyen
    observation, reward, terminated, truncated, info = env.step(1)
    print()
    print(f"After action 1 (DOWN) -> state {observation}, reward {reward}")
    print(env.render())

    env.close()


if __name__ == "__main__":
    main()
