"""Bai 12 - Xay dung mot MDP nho hai state, hai action.

Bieu dien giong dinh dang cua Gymnasium:
    P[state][action] = [(probability, next_state, reward, terminated), ...]

Cau chuyen: State 0 = "Dang lam viec", State 1 = "Dang nghi".
    Action 0 = "Tiep tuc lam", Action 1 = "Doi trang thai".
"""

N_STATES = 2
N_ACTIONS = 2

STATE_NAMES = {0: "WORKING", 1: "RESTING"}
ACTION_NAMES = {0: "STAY", 1: "SWITCH"}

# Tong xac suat cua moi cap (state, action) phai bang 1
P = {
    0: {
        # O state 0, STAY: 80% o lai state 0 (reward +2), 20% truot sang state 1
        0: [
            (0.8, 0, 2.0, False),
            (0.2, 1, 0.0, False),
        ],
        # O state 0, SWITCH: 90% sang state 1 (reward -1), 10% that bai o lai
        1: [
            (0.9, 1, -1.0, False),
            (0.1, 0, 0.0, False),
        ],
    },
    1: {
        # O state 1, STAY: 70% o lai (reward +1), 30% ket thuc episode (reward +5)
        0: [
            (0.7, 1, 1.0, False),
            (0.3, 1, 5.0, True),
        ],
        # O state 1, SWITCH: chac chan quay ve state 0 voi reward 0
        1: [
            (1.0, 0, 0.0, False),
        ],
    },
}


def print_mdp(P) -> None:
    """In toan bo model chuyen trang thai duoi dang bang."""
    print("state | action | prob | next_state | reward | terminated")
    print("------+--------+------+------------+--------+-----------")
    for state in sorted(P):
        for action in sorted(P[state]):
            for probability, next_state, reward, terminated in P[state][action]:
                print(f"{state:^6d}| {action:^7d}| {probability:^5.2f}| "
                      f"{next_state:^11d}| {reward:^7.1f}| {str(terminated):^10s}")


def main() -> None:
    print("MDP hai state, hai action")
    print()
    print("States :", STATE_NAMES)
    print("Actions:", ACTION_NAMES)
    print()
    print_mdp(P)

    print()
    print("Tong xac suat cua tung cap (state, action):")
    for state in sorted(P):
        for action in sorted(P[state]):
            total = sum(transition[0] for transition in P[state][action])
            print(f"  P[{state}][{action}] -> {total:.4f}")


if __name__ == "__main__":
    main()
