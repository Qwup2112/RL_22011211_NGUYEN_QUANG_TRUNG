"""Bai 20 - So sanh FrozenLake deterministic va stochastic tai cung mot (s, a)."""

from mdp_utils import ACTION_NAMES, create_environment, get_transition_model

STATE = 0
ACTION = 2   # RIGHT


def show_transitions(is_slippery: bool, state: int, action: int) -> int:
    """In transition cua (state, action) trong mot cau hinh, tra ve so transition."""
    env = create_environment(map_name="4x4", is_slippery=is_slippery)
    transitions = get_transition_model(env)[state][action]

    print(f"is_slippery={is_slippery}: state={state}, "
          f"action={action} ({ACTION_NAMES[action]})")
    print(f"  So transition: {len(transitions)}")
    for probability, next_state, reward, terminated in transitions:
        print(f"    p={probability:.6f} -> next_state={next_state}, "
              f"reward={reward}, terminated={terminated}")
    print(f"  Tong xac suat: {sum(t[0] for t in transitions):.6f}")
    print()

    env.close()
    return len(transitions)


def main() -> None:
    n_deterministic = show_transitions(False, STATE, ACTION)
    n_stochastic = show_transitions(True, STATE, ACTION)

    print(f"So transition: deterministic = {n_deterministic}, "
          f"stochastic = {n_stochastic}")
    print()

    # KET LUAN (3-5 dong):
    # 1. Voi is_slippery=False, moi (state, action) chi co DUNG MOT transition
    #    xac suat 1.0: action RIGHT tai state 0 chac chan dua agent sang state 1.
    # 2. Voi is_slippery=True, cung (state, action) do co BA transition, moi
    #    huong xac suat 1/3: di dung huong (state 1) hoac truot sang hai huong
    #    vuong goc (state 4 va state 0 - dung tuong nen o nguyen).
    # 3. Vi vay Bellman backup bat buoc phai cong theo xac suat cua tung
    #    transition; neu chi lay transition dau tien thi value function sai hoan toan.
    # 4. Tinh ngau nhien nay lam bai toan kho hon: policy toi uu khong con la
    #    "di duong ngan nhat" ma phai tranh xa cac o ho de giam rui ro bi truot.
    print("Ket luan: xem comment trong file - deterministic co 1 transition,")
    print("stochastic co 3 transition moi cai xac suat 1/3, nen Bellman backup")
    print("phai duyet va cong theo xac suat cua tat ca transition.")


if __name__ == "__main__":
    main()
