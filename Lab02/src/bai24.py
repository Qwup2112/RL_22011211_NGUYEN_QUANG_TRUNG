"""Bai 24 - Iterative Policy Evaluation.

Lap Bellman expectation backup cho den khi delta < theta.
Ham policy_evaluation() nam trong src/mdp_utils.py, tra ve (V, n_iterations).
"""

import numpy as np

from mdp_utils import (create_environment, env_sizes, policy_evaluation,
                       print_value_grid, uniform_random_policy)


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    n_states, n_actions = env_sizes(env)

    policy = uniform_random_policy(n_states, n_actions)
    gamma = 0.99
    theta = 1e-8

    V, n_iterations = policy_evaluation(env, policy, gamma=gamma, theta=theta)

    print(f"Iterative Policy Evaluation cho uniform random policy")
    print(f"gamma = {gamma}, theta = {theta}")
    print(f"Hoi tu sau {n_iterations} iteration")
    print()
    print("V_pi (dang luoi 4x4):")
    print_value_grid(env, V)
    print()
    print("V_pi (vector):", np.round(V, 6))
    print()
    print(f"V(state 0) = {V[0]:.6f}  -> xac suat thang cua random policy neu")
    print("gamma gan 1, vi reward chi bang 1 khi toi Goal.")

    # Anh huong cua theta len so iteration
    print()
    print("Anh huong cua theta:")
    print("  theta  | iterations | V(0)")
    print("---------+------------+-----------")
    for theta_value in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
        V_test, n_test = policy_evaluation(env, policy, gamma=gamma, theta=theta_value)
        print(f"  {theta_value:<7.0e}| {n_test:^11d}| {V_test[0]:.8f}")

    env.close()


if __name__ == "__main__":
    main()
