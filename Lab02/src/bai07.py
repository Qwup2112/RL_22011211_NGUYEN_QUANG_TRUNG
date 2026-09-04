"""Bai 7 - Undiscounted return.

G_t = R_(t+1) + gamma*R_(t+2) + gamma^2*R_(t+3) + ...
Voi gamma = 1.0 thi return chinh la tong reward.
"""


def compute_return(rewards, gamma: float) -> float:
    """Tinh discounted return G_0 tu mot chuoi reward."""
    total = 0.0
    discount = 1.0

    for reward in rewards:
        total += discount * float(reward)
        discount *= gamma

    return total


def main() -> None:
    rewards = [1, 1, 1, 1, 1]

    gamma = 1.0
    G0 = compute_return(rewards, gamma)

    print("rewards =", rewards)
    print("gamma   =", gamma)
    print(f"G_0     = {G0}")
    print()
    print("Kiem tra: voi gamma = 1.0 thi G_0 = sum(rewards) =", sum(rewards))
    print("Khop:", G0 == float(sum(rewards)))


if __name__ == "__main__":
    main()
