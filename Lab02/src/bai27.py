"""Bai 27 - Hien thi policy tren luoi 4x4 bang mui ten.

Ky hieu: LEFT, DOWN, RIGHT, UP. O Hole in H, o Goal in G.
Ham print_policy_grid() nam trong src/mdp_utils.py va tu dong chuyen sang
ky hieu ASCII neu terminal khong hien thi duoc Unicode.
"""

from mdp_utils import (ACTION_NAMES, ACTION_SYMBOLS, action_symbols,
                       create_environment, env_sizes,
                       greedy_policy_from_value, policy_evaluation,
                       print_policy_grid, uniform_random_policy)


def print_frozenlake_policy(env, policy) -> None:
    """In policy cua FrozenLake duoi dang luoi (goi lai ham dung chung)."""
    print_policy_grid(env, policy)


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    n_states, n_actions = env_sizes(env)
    gamma = 0.99

    symbols = action_symbols()
    print("Bang ky hieu action:")
    for action in sorted(ACTION_SYMBOLS):
        print(f"  {action} = {ACTION_NAMES[action]:<5s} -> {symbols[action]}")
    print()

    desc = env.unwrapped.desc
    print("Ban do:")
    for row in desc:
        print("  " + " ".join(cell.decode("ascii") for cell in row))
    print()

    random_policy = uniform_random_policy(n_states, n_actions)
    V, _ = policy_evaluation(env, random_policy, gamma=gamma)
    greedy_policy = greedy_policy_from_value(env, V, gamma)

    print("Greedy policy tu V cua random policy:")
    print_frozenlake_policy(env, greedy_policy)
    print()
    print("policy (vector):", greedy_policy)

    env.close()


if __name__ == "__main__":
    main()
