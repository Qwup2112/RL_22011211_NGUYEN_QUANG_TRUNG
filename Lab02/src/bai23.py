"""Bai 23 - Mot sweep cua Policy Evaluation voi uniform random policy.

    V_pi(s) <- sum_a pi(a|s) * sum_(s') p(s'|s,a) * [r + gamma * V_pi(s')]

Ham policy_evaluation_sweep() nam trong src/mdp_utils.py.
"""

import numpy as np

from mdp_utils import (create_environment, env_sizes, policy_evaluation_sweep,
                       print_value_grid, uniform_random_policy)


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    n_states, n_actions = env_sizes(env)
    gamma = 0.99

    policy = uniform_random_policy(n_states, n_actions)
    print("Uniform random policy, pi(a|s) =", 1 / n_actions, "cho moi action")
    print()

    V = np.zeros(n_states)
    print("V truoc sweep:")
    print_value_grid(env, V)

    # Chi thuc hien DUNG MOT sweep qua tat ca state
    V_after_one = policy_evaluation_sweep(env, policy, V, gamma)

    print()
    print("V sau 1 sweep:")
    print_value_grid(env, V_after_one)
    print()
    print("delta sau sweep 1 =", float(np.max(np.abs(V_after_one - V))))

    # Vai sweep tiep theo de thay gia tri lan truyen nguoc tu Goal
    V_current = V_after_one
    for sweep in range(2, 6):
        V_next = policy_evaluation_sweep(env, policy, V_current, gamma)
        delta = float(np.max(np.abs(V_next - V_current)))
        print()
        print(f"V sau {sweep} sweep (delta = {delta:.8f}):")
        print_value_grid(env, V_next)
        V_current = V_next

    print()
    print("Nhan xet: sau sweep dau tien chi state 14 co gia tri khac 0 (no canh")
    print("Goal). Moi sweep tiep theo gia tri lan truyen nguoc them mot o - day")
    print("chinh la cach thong tin ve reward lan tu Goal ve Start.")

    env.close()


if __name__ == "__main__":
    main()
