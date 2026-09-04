"""Bai 17 - In toan bo transition model cua mot state."""

from mdp_utils import ACTION_NAMES, create_environment, get_transition_model


def print_state_transitions(env, state: int) -> None:
    """In moi transition cua tat ca action tai mot state."""
    transition_model = get_transition_model(env)

    print(f"Transition model cua state {state}:")
    print()
    print("Action | Name  | Probability | Next state | Reward | Terminated")
    print("-------+-------+-------------+------------+--------+-----------")
    for action in sorted(transition_model[state]):
        for probability, next_state, reward, terminated in transition_model[state][action]:
            print(f"{action:^7d}| {ACTION_NAMES[action]:<6s}| {probability:^12.6f}| "
                  f"{next_state:^11d}| {reward:^7.1f}| {str(terminated):^10s}")


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)

    print_state_transitions(env, 0)

    print()
    print("Moi action co 3 transition vi is_slippery=True: agent di dung huong")
    print("voi xac suat 1/3, va truot sang hai huong vuong goc moi huong 1/3.")
    print("Khong duoc gia dinh moi (state, action) chi co mot next_state.")

    env.close()


if __name__ == "__main__":
    main()
