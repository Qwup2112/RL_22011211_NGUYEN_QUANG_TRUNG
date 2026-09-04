"""Bai 19 - Kiem tra tong xac suat transition cua moi (state, action)."""

import numpy as np

from mdp_utils import create_environment, env_sizes, get_transition_model


def check_all_transitions(env, tol: float = 1e-8) -> tuple[bool, int, int]:
    """Duyet moi (state, action) va xac nhan tong xac suat bang 1.

    Tra ve (tat_ca_hop_le, so_cap_kiem_tra, so_cap_loi).
    """
    transition_model = get_transition_model(env)
    n_states, n_actions = env_sizes(env)

    n_checked = 0
    n_invalid = 0

    for state in range(n_states):
        for action in range(n_actions):
            probabilities = [transition[0] for transition in transition_model[state][action]]
            n_checked += 1

            if not np.isclose(sum(probabilities), 1.0, atol=tol):
                n_invalid += 1
                print(f"Invalid transition at state={state}, action={action}: "
                      f"tong = {sum(probabilities)}")

    return n_invalid == 0, n_checked, n_invalid


def transition_count_table(env) -> dict[int, int]:
    """Dem so cap (state, action) theo so luong transition cua chung."""
    transition_model = get_transition_model(env)
    n_states, n_actions = env_sizes(env)

    counts: dict[int, int] = {}
    for state in range(n_states):
        for action in range(n_actions):
            size = len(transition_model[state][action])
            counts[size] = counts.get(size, 0) + 1

    return counts


def main() -> None:
    for is_slippery in [False, True]:
        env = create_environment(map_name="4x4", is_slippery=is_slippery)

        print(f"=== is_slippery={is_slippery} ===")
        valid, n_checked, n_invalid = check_all_transitions(env)
        print(f"Da kiem tra {n_checked} cap (state, action)")
        print(f"So cap khong hop le: {n_invalid}")
        print(f"Tat ca hop le: {valid}")

        counts = transition_count_table(env)
        print("Phan bo so transition tren moi cap (state, action):")
        for size in sorted(counts):
            print(f"  {size} transition: {counts[size]} cap")
        print()

        env.close()


if __name__ == "__main__":
    main()
