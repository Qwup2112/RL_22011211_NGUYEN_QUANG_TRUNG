"""Bai 30 - Kiem tra tinh on dinh cua policy trong Policy Iteration.

Bien policy_stable duoc tu lap trinh (so sanh policy cu va policy moi),
khong dung ham co san cua thu vien nao.
"""

import numpy as np

from mdp_utils import (create_environment, env_sizes,
                       greedy_policy_from_value, policy_evaluation,
                       print_policy_grid)


def policy_iteration_verbose(env, gamma: float = 0.99, theta: float = 1e-8,
                             max_iterations: int = 1000):
    """Policy Iteration co in ro tung buoc va bien policy_stable."""
    n_states, _ = env_sizes(env)

    policy = np.zeros(n_states, dtype=np.int64)
    V = np.zeros(n_states, dtype=np.float64)
    n_iterations = 0

    for iteration in range(1, max_iterations + 1):
        V, eval_iterations = policy_evaluation(env, policy, gamma, theta)
        new_policy = greedy_policy_from_value(env, V, gamma)

        # Tu lap trinh kiem tra on dinh: policy on dinh khi khong state nao doi action
        policy_stable = bool(np.all(new_policy == policy))
        n_changed = int(np.sum(new_policy != policy))

        print(f"Iteration {iteration:2d}: eval_sweeps={eval_iterations:4d}, "
              f"changed_states={n_changed:2d}, policy_stable={policy_stable}")

        policy = new_policy
        n_iterations = iteration

        if policy_stable:
            break

    return policy, V, n_iterations


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    gamma = 0.99

    policy, V, n_iterations = policy_iteration_verbose(env, gamma=gamma)

    print()
    print(f"Policy Iteration converged after {n_iterations} iterations.")
    print()
    print("Optimal policy:")
    print_policy_grid(env, policy)

    # Kiem chung: policy toi uu phai la greedy doi voi chinh V cua no
    V_final, _ = policy_evaluation(env, policy, gamma=gamma)
    greedy_again = greedy_policy_from_value(env, V_final, gamma)
    print()
    print("Kiem chung dieu kien toi uu (policy la greedy voi V cua chinh no):",
          bool(np.array_equal(policy, greedy_again)))
    print("Sai lech V lon nhat giua hai lan danh gia:",
          float(np.max(np.abs(V - V_final))))

    env.close()


if __name__ == "__main__":
    main()
