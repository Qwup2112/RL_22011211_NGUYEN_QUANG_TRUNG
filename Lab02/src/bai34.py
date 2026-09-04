"""Bai 34 - Danh gia policy bang simulation that tren moi truong.

Ham evaluate_policy_by_simulation() nam trong src/mdp_utils.py; action duoc
lay bang action = policy[state].
"""

import numpy as np

from mdp_utils import (create_environment, env_sizes,
                       evaluate_policy_by_simulation,
                       greedy_policy_from_value, policy_iteration,
                       print_policy_grid, value_iteration)


def random_deterministic_policy(n_states: int, n_actions: int,
                                seed: int = 0) -> np.ndarray:
    """Policy ngau nhien nhung co dinh: moi state gan mot action ngau nhien."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_actions, size=n_states).astype(np.int64)


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    n_states, n_actions = env_sizes(env)
    gamma = 0.99
    n_episodes = 1000

    # 1) Random policy
    random_policy = random_deterministic_policy(n_states, n_actions, seed=0)

    # 2) Policy tu Value Iteration
    V_vi, n_vi, _ = value_iteration(env, gamma=gamma, theta=1e-8)
    policy_vi = greedy_policy_from_value(env, V_vi, gamma)

    # 3) Policy tu Policy Iteration
    policy_pi, V_pi, n_pi = policy_iteration(env, gamma=gamma, theta=1e-8)

    policies = {
        "Random": random_policy,
        "Value Iteration": policy_vi,
        "Policy Iteration": policy_pi,
    }

    print(f"Danh gia bang mo phong: {n_episodes} episode moi policy, seed = 42")
    print()
    print("Policy           | Success | Success rate | Mean reward | Mean len | Min | Max")
    print("-----------------+---------+--------------+-------------+----------+-----+-----")
    results = {}
    for name, policy in policies.items():
        result = evaluate_policy_by_simulation(env, policy,
                                               n_episodes=n_episodes, seed=42)
        results[name] = result
        print(f"{name:<17s}| {result['success']:^8d}| {result['success_rate']:^13.4f}| "
              f"{result['mean_reward']:^12.4f}| {result['mean_length']:^9.2f}| "
              f"{result['min_length']:^4d}| {result['max_length']:^4d}")

    print()
    print("Policy tu Value Iteration:")
    print_policy_grid(env, policy_vi)
    print()
    print("Hai policy DP giong nhau:", bool(np.array_equal(policy_vi, policy_pi)))
    print(f"V*(state 0) tu ly thuyet     : {V_vi[0]:.4f}")
    print(f"Success rate do bang mo phong: {results['Value Iteration']['success_rate']:.4f}")
    print()
    print("Hai so tren khong bang nhau vi V* la return CO CHIET KHAU (gamma=0.99)")
    print("con success rate la ty le thang khong chiet khau; V* luon nho hon.")

    env.close()


if __name__ == "__main__":
    main()
