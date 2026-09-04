"""Bai 13 - Kiem tra tinh hop le cua mot model MDP."""

import numpy as np

from bai12 import N_ACTIONS, N_STATES, P


def validate_mdp(P, n_states: int, n_actions: int, tol: float = 1e-8) -> bool:
    """Kiem tra tong xac suat transition cua tung (state, action) bang 1.

    In thong bao khi phat hien cap (state, action) khong hop le.
    """
    valid = True

    for state in range(n_states):
        for action in range(n_actions):
            if state not in P or action not in P[state]:
                print(f"Invalid transition at state={state}, action={action} "
                      f"(khong co trong model)")
                valid = False
                continue

            transitions = P[state][action]
            total = sum(float(transition[0]) for transition in transitions)

            if not np.isclose(total, 1.0, atol=tol):
                print(f"Invalid transition at state={state}, action={action} "
                      f"(tong xac suat = {total})")
                valid = False

            for probability, next_state, reward, terminated in transitions:
                if not 0.0 <= probability <= 1.0:
                    print(f"Invalid transition at state={state}, action={action} "
                          f"(xac suat {probability} ngoai [0,1])")
                    valid = False
                if not 0 <= int(next_state) < n_states:
                    print(f"Invalid transition at state={state}, action={action} "
                          f"(next_state {next_state} khong hop le)")
                    valid = False

    return valid


def main() -> None:
    print("Kiem tra MDP cua Bai 12:")
    print("Ket qua:", validate_mdp(P, N_STATES, N_ACTIONS))
    print()

    # Tao mot ban sao bi loi de kiem thu thong bao
    broken = {state: {action: list(P[state][action]) for action in P[state]}
              for state in P}
    broken[0][1] = [(0.5, 1, -1.0, False), (0.2, 0, 0.0, False)]

    print("Kiem tra MDP co loi (tong xac suat P[0][1] = 0.7):")
    print("Ket qua:", validate_mdp(broken, N_STATES, N_ACTIONS))


if __name__ == "__main__":
    main()
