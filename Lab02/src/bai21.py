"""Bai 21 - Mot Bellman backup: Q(s,a) tu V.

    Q(s,a) = sum_(s') p(s'|s,a) * [r + gamma * V(s')]

Ham q_from_v() duoc cai dat trong src/mdp_utils.py (muc 14 cua de bai yeu cau
gom cac ham dung chung vao do). File nay minh hoa va KIEM CHUNG lai ket qua
bang mot vong lap tinh tay tach bach tung so hang.
"""

import numpy as np

from mdp_utils import (ACTION_NAMES, create_environment, env_sizes,
                       get_transition_model, q_from_v)


def q_from_v_step_by_step(env, V, state: int, action: int, gamma: float) -> float:
    """Tinh Q(s,a) va in ro tung so hang, dung de doi chieu voi q_from_v()."""
    transitions = get_transition_model(env)[state][action]

    print(f"Q({state}, {action}) voi gamma = {gamma}:")
    total = 0.0
    for probability, next_state, reward, terminated in transitions:
        # Neu transition dan toi trang thai ket thuc thi khong bootstrap tiep
        future_value = 0.0 if terminated else gamma * V[next_state]
        term = probability * (reward + future_value)
        total += term

        print(f"  {probability:.6f} * ({reward} + "
              f"{'0 (terminated)' if terminated else f'{gamma} * V[{next_state}]={future_value:.6f}'})"
              f" = {term:.6f}")

    print(f"  => Q({state}, {action}) = {total:.6f}")
    return total


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)
    n_states, n_actions = env_sizes(env)
    gamma = 0.99

    # V ban dau bang 0 -> Q chi con phan expected immediate reward
    V = np.zeros(n_states)

    print("--- V = 0, state 14 (o ben trai Goal) ---")
    for action in range(n_actions):
        manual = q_from_v_step_by_step(env, V, 14, action, gamma)
        library = q_from_v(env, V, 14, action, gamma)
        print(f"  q_from_v() = {library:.6f}  |  khop: {np.isclose(manual, library)}")
        print()

    # Voi mot V khac 0 de thay ro phan bootstrap
    V = np.linspace(0.0, 1.0, n_states)
    print("--- V = linspace(0, 1, 16), state 0, action RIGHT ---")
    manual = q_from_v_step_by_step(env, V, 0, 2, gamma)
    library = q_from_v(env, V, 0, 2, gamma)
    print(f"  q_from_v() = {library:.6f}  |  khop: {np.isclose(manual, library)}")

    print()
    print("Q cua state 14 voi V = linspace:")
    for action in range(n_actions):
        value = q_from_v(env, V, 14, action, gamma)
        print(f"  action {action} ({ACTION_NAMES[action]:<5s}): Q = {value:.6f}")

    env.close()


if __name__ == "__main__":
    main()
