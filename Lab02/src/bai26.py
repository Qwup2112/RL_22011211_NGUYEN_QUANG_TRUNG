"""Bai 26 - Greedy policy tu value function (Policy Improvement).

Voi moi state: tinh Q(s,a) cho moi action, chon np.argmax(q_values).
Ham greedy_policy_from_value() nam trong src/mdp_utils.py.
"""

import numpy as np

from mdp_utils import (ACTION_NAMES, action_values, create_environment,
                       env_sizes, greedy_policy_from_value, policy_evaluation,
                       uniform_random_policy)


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    n_states, n_actions = env_sizes(env)
    gamma = 0.99

    # 1) Tu V = 0: moi Q deu bang nhau (tru state 14) nen argmax tra ve action 0
    V_zero = np.zeros(n_states)
    policy_from_zero = greedy_policy_from_value(env, V_zero, gamma)
    print("Greedy policy tu V = 0:")
    print(" ", policy_from_zero)
    print("  (V = 0 khong chua thong tin nen argmax gan nhu luon chon action 0)")
    print()

    # 2) Tu V cua uniform random policy
    random_policy = uniform_random_policy(n_states, n_actions)
    V_random, n_iterations = policy_evaluation(env, random_policy, gamma=gamma)
    greedy_policy = greedy_policy_from_value(env, V_random, gamma)

    print(f"V cua random policy (hoi tu sau {n_iterations} iteration)")
    print("Greedy policy tu V do:")
    print(" ", greedy_policy)
    print()

    print("state | Q(s,LEFT) | Q(s,DOWN) | Q(s,RIGHT)|  Q(s,UP)  | argmax")
    print("------+-----------+-----------+-----------+-----------+--------")
    for state in range(n_states):
        q_values = action_values(env, V_random, state, gamma)
        row = " | ".join(f"{value:^10.6f}" for value in q_values)
        best = int(np.argmax(q_values))
        print(f"{state:^6d}| {row}| {ACTION_NAMES[best]}")

    print()
    print("So state ma greedy policy khac random-argmax:",
          int(np.sum(greedy_policy != policy_from_zero)))

    env.close()


if __name__ == "__main__":
    main()
