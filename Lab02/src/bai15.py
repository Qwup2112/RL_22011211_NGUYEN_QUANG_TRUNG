"""Bai 15 - Stochastic policy: moi state la mot phan phoi xac suat tren action."""

import numpy as np

from bai12 import N_ACTIONS, N_STATES


def uniform_random_policy(n_states: int, n_actions: int) -> np.ndarray:
    """Policy chon deu moi action tai moi state."""
    return np.ones((n_states, n_actions)) / n_actions


def validate_stochastic_policy(policy, tol: float = 1e-10) -> bool:
    """Kiem tra tong xac suat action tai moi state bang 1."""
    policy_array = np.asarray(policy, dtype=np.float64)

    if policy_array.ndim != 2:
        print("Policy stochastic phai la ma tran 2 chieu (n_states, n_actions)")
        return False

    if np.any(policy_array < -tol) or np.any(policy_array > 1.0 + tol):
        print("Co xac suat nam ngoai [0, 1]")
        return False

    row_sums = policy_array.sum(axis=1)
    ok = True
    for state, total in enumerate(row_sums):
        if not np.isclose(total, 1.0, atol=max(tol, 1e-12)):
            print(f"Invalid policy at state={state} (tong xac suat = {total})")
            ok = False

    return ok


def main() -> None:
    policy = uniform_random_policy(N_STATES, N_ACTIONS)

    print("Uniform random policy:")
    print(policy)
    print("shape =", policy.shape)
    print()
    print("Tong xac suat action tai moi state:", policy.sum(axis=1))
    print("Hop le:", validate_stochastic_policy(policy))
    print()

    # Mot stochastic policy khac: thien ve action 0
    biased = np.array([
        [0.7, 0.3],
        [0.1, 0.9],
    ])
    print("Stochastic policy thien lech:")
    print(biased)
    print("Hop le:", validate_stochastic_policy(biased))
    print()

    broken = np.array([
        [0.7, 0.5],
        [0.1, 0.9],
    ])
    print("Policy sai (state 0 tong = 1.2):")
    print("Hop le:", validate_stochastic_policy(broken))
    print()

    # Deterministic policy chinh la truong hop dac biet cua stochastic policy
    deterministic = np.array([0, 1])
    one_hot = np.zeros((N_STATES, N_ACTIONS))
    one_hot[np.arange(N_STATES), deterministic] = 1.0
    print("Deterministic policy viet duoi dang stochastic (one-hot):")
    print(one_hot)
    print("Hop le:", validate_stochastic_policy(one_hot))


if __name__ == "__main__":
    main()
