"""Bai 9 - Tinh return cho moi buoc, duyet nguoc tu cuoi episode."""


def discounted_returns(rewards, gamma: float) -> list[float]:
    """Tra ve [G_0, G_1, ..., G_{T-1}] tinh theo chieu tu cuoi episode ve dau.

    Cong thuc de quy: G_t = R_(t+1) + gamma * G_(t+1), voi G_T = 0.
    Cach nay chi mat O(T) thay vi O(T^2) neu tinh lai tu dau moi buoc.
    """
    returns = [0.0] * len(rewards)
    running_return = 0.0

    for index in reversed(range(len(rewards))):
        running_return = float(rewards[index]) + gamma * running_return
        returns[index] = running_return

    return returns


def main() -> None:
    rewards = [0, 0, 0, 1]

    for gamma in [1.0, 0.9, 0.5]:
        returns = discounted_returns(rewards, gamma)
        print(f"gamma = {gamma}")
        for index, value in enumerate(returns):
            print(f"  G_{index} = {value:.6f}")
        print()

    # Kiem tra lai bang cach tinh truc tiep G_0
    from bai07 import compute_return
    gamma = 0.9
    print("Kiem tra G_0 (gamma = 0.9):")
    print("  discounted_returns:", discounted_returns(rewards, gamma)[0])
    print("  compute_return    :", compute_return(rewards, gamma))


if __name__ == "__main__":
    main()
