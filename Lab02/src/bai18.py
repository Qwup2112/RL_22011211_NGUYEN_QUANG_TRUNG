"""Bai 18 - Ham describe_state() mo ta day du mot state cua FrozenLake."""

from mdp_utils import ACTION_NAMES, create_environment, get_transition_model

CELL_MEANING = {
    "S": "Start",
    "F": "Frozen (di duoc)",
    "H": "Hole (roi xuong la ket thuc)",
    "G": "Goal (dich, reward 1)",
}


def describe_state(env, state: int) -> None:
    """In vi tri, loai o va toan bo transition cua mot state."""
    desc = env.unwrapped.desc
    n_cols = desc.shape[1]
    row, col = divmod(int(state), n_cols)
    cell_type = desc[row][col].decode("ascii")

    print(f"=== State {state} ===")
    print(f"  Vi tri     : row {row}, col {col}")
    print(f"  Loai o     : {cell_type} - {CELL_MEANING[cell_type]}")

    transition_model = get_transition_model(env)
    is_terminal = all(
        terminated
        for action in transition_model[state]
        for _, _, _, terminated in transition_model[state][action]
    )
    print(f"  Terminal   : {is_terminal}")

    for action in sorted(transition_model[state]):
        transitions = transition_model[state][action]
        expected_reward = sum(p * r for p, _, r, _ in transitions)
        print(f"  Action {action} ({ACTION_NAMES[action]:<5s}): "
              f"{len(transitions)} transition, E[reward] = {expected_reward:.4f}")
        for probability, next_state, reward, terminated in transitions:
            print(f"      p={probability:.4f} -> state {next_state:2d}, "
                  f"reward={reward}, terminated={terminated}")
    print()


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)

    for state in [0, 1, 14]:
        describe_state(env, state)

    print("State 14 la o ngay ben trai Goal: action RIGHT co 1/3 co hoi ket thuc")
    print("voi reward = 1, day la state co gia tri cao nhat trong ban do.")

    env.close()


if __name__ == "__main__":
    main()
