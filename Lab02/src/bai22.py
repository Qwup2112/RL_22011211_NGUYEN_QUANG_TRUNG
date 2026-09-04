"""Bai 22 - Tinh toan bo vector Q(s, .) cua mot state.

Ham action_values() nam trong src/mdp_utils.py va tra ve vector do dai
env.action_space.n.
"""

import numpy as np

from mdp_utils import (ACTION_NAMES, action_values, create_environment,
                       env_sizes)


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    n_states, n_actions = env_sizes(env)
    gamma = 0.99

    V = np.zeros(n_states)

    print(f"Kiem thu voi V = zeros({n_states}), gamma = {gamma}")
    print()
    header = "state | " + " | ".join(f"Q(s,{a}) {ACTION_NAMES[a][:1]}" for a in range(n_actions))
    print(header)
    print("-" * len(header))
    for state in range(n_states):
        q_values = action_values(env, V, state, gamma)
        assert q_values.shape == (n_actions,), "Vector Q phai co do dai n_actions"
        row = " | ".join(f"{value:^9.6f}" for value in q_values)
        print(f"{state:^6d}| {row}")

    positive_states = [state for state in range(n_states)
                       if float(np.max(action_values(env, V, state, gamma))) > 0]
    print()
    print("Voi V = 0, Q(s,a) chi con expected immediate reward.")
    print(f"Cac state co Q duong: {positive_states} - do la state duy nhat ma tu do")
    print("agent co the buoc thang vao Goal va nhan reward = 1 ngay lap tuc.")

    q_state_14 = action_values(env, V, 14, gamma)
    print()
    print("Q(14, .) =", np.round(q_state_14, 6))
    print("shape    =", q_state_14.shape)
    print("argmax   =", int(np.argmax(q_state_14)),
          f"({ACTION_NAMES[int(np.argmax(q_state_14))]})")

    env.close()


if __name__ == "__main__":
    main()
