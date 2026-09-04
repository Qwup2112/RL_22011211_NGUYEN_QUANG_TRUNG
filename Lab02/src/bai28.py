"""Bai 28 - Mot buoc Policy Improvement day du.

1. Policy Evaluation cho policy hien tai;
2. Greedy Policy Improvement;
3. so sanh old_policy va new_policy;
4. dem so state doi action.
"""

import numpy as np

from mdp_utils import (ACTION_NAMES, create_environment, env_sizes,
                       greedy_policy_from_value, policy_evaluation,
                       print_policy_grid, print_value_grid)


def improvement_step(env, policy, gamma: float = 0.99, theta: float = 1e-8):
    """Mot vong Policy Evaluation + Policy Improvement.

    Tra ve (new_policy, V, n_eval_iterations, changed_states).
    """
    V, n_eval_iterations = policy_evaluation(env, policy, gamma=gamma, theta=theta)
    new_policy = greedy_policy_from_value(env, V, gamma)

    old_array = np.asarray(policy)
    if old_array.ndim == 2:
        old_array = np.argmax(old_array, axis=1)

    changed_states = np.flatnonzero(new_policy != old_array)
    return new_policy, V, n_eval_iterations, changed_states


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    n_states, n_actions = env_sizes(env)
    gamma = 0.99

    # Policy ban dau: luon di LEFT
    policy = np.zeros(n_states, dtype=np.int64)

    # Bai tap yeu cau MOT buoc; chay them vai buoc de thay policy on dinh dan
    for step in range(1, 11):
        old_policy = policy.copy()
        policy, V, n_eval, changed = improvement_step(env, policy, gamma)

        print(f"=== Buoc Policy Improvement thu {step} ===")
        print(f"Policy Evaluation hoi tu sau {n_eval} iteration")
        print("V cua policy cu:")
        print_value_grid(env, V)
        print("Policy cu   :", old_policy)
        print("Policy moi  :", policy)
        print(f"So state doi action: {len(changed)}")
        for state in changed:
            print(f"    state {int(state):2d}: {ACTION_NAMES[int(old_policy[state])]:<5s}"
                  f" -> {ACTION_NAMES[int(policy[state])]}")
        print("Policy moi dang luoi:")
        print_policy_grid(env, policy)
        print()

        if len(changed) == 0:
            print(f"Policy khong doi nua => da on dinh sau {step} buoc.")
            break

    env.close()


if __name__ == "__main__":
    main()
