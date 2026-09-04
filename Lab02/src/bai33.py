"""Bai 33 - Trich xuat optimal policy sau Value Iteration."""

import numpy as np

from mdp_utils import (ACTION_NAMES, action_values, create_environment,
                       env_sizes, greedy_policy_from_value, print_policy_grid,
                       print_value_grid, value_iteration)


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    n_states, n_actions = env_sizes(env)
    gamma = 0.99

    V, n_iterations, deltas = value_iteration(env, gamma=gamma, theta=1e-8)
    optimal_policy = greedy_policy_from_value(env, V, gamma)

    print(f"Value Iteration hoi tu sau {n_iterations} iteration")
    print()
    print("Optimal state values:")
    print_value_grid(env, V)
    print()
    print("Optimal policy (dang luoi 4x4):")
    print_policy_grid(env, optimal_policy)
    print()
    print("Optimal policy (vector):", optimal_policy)
    print()

    print("state | V*(s)    | action toi uu | Q*(s,a)")
    print("------+----------+---------------+---------------------------------")
    for state in range(n_states):
        q_values = action_values(env, V, state, gamma)
        best = int(optimal_policy[state])
        q_text = " ".join(f"{value:7.4f}" for value in q_values)
        print(f"{state:^6d}| {V[state]:^9.6f}| {ACTION_NAMES[best]:<14s}| {q_text}")

    print()
    print("Kiem chung V*(s) = max_a Q*(s,a):",
          bool(np.allclose(V, [np.max(action_values(env, V, s, gamma))
                               for s in range(n_states)], atol=1e-6)))

    print()
    print("Nhan xet: policy toi uu KHONG di duong ngan nhat. Vi du o state 0 no")
    print("chon LEFT - dung vao tuong. Do la cach an toan nhat: khi truot, agent")
    print("chi co the di len hoac xuong chu khong the roi vao ho o state 5.")

    env.close()


if __name__ == "__main__":
    main()
