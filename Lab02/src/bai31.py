"""Bai 31 - Mot sweep cua Value Iteration.

    new_V[s] = max_a Q(s,a)      (Bellman optimality backup)

Khac Policy Evaluation o cho lay max thay vi lay trung binh theo policy.
Ham value_iteration_sweep() nam trong src/mdp_utils.py.
"""

import numpy as np

from mdp_utils import (create_environment, env_sizes, policy_evaluation_sweep,
                       print_value_grid, uniform_random_policy,
                       value_iteration_sweep)


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    n_states, n_actions = env_sizes(env)
    gamma = 0.99

    V = np.zeros(n_states)
    print("V ban dau:")
    print_value_grid(env, V)

    for sweep in range(1, 6):
        V_next = value_iteration_sweep(env, V, gamma)
        delta = float(np.max(np.abs(V_next - V)))
        print()
        print(f"Sau {sweep} sweep cua Value Iteration (delta = {delta:.8f}):")
        print_value_grid(env, V_next)
        V = V_next

    # So sanh voi Policy Evaluation sweep cua uniform random policy
    random_policy = uniform_random_policy(n_states, n_actions)
    V_pe = np.zeros(n_states)
    V_vi = np.zeros(n_states)
    for _ in range(5):
        V_pe = policy_evaluation_sweep(env, random_policy, V_pe, gamma)
        V_vi = value_iteration_sweep(env, V_vi, gamma)

    print()
    print("Sau 5 sweep, so sanh hai loai backup:")
    print("  Policy Evaluation (trung binh theo pi):", np.round(V_pe[:8], 6))
    print("  Value Iteration   (lay max theo a)    :", np.round(V_vi[:8], 6))
    print()
    print("Value Iteration cho gia tri lon hon o moi state vi no gia dinh agent")
    print("luon chon action tot nhat, thay vi chon deu nhu random policy.")

    env.close()


if __name__ == "__main__":
    main()
