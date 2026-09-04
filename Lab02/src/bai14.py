"""Bai 14 - Deterministic policy: moi state ung voi dung mot action."""

import numpy as np

from bai12 import ACTION_NAMES, N_STATES, STATE_NAMES

# policy[state] = action se chon tai state do
policy = np.array([0, 1])   # WORKING -> STAY, RESTING -> SWITCH


def print_policy(policy, state_names=None, action_names=None) -> None:
    """In deterministic policy duoi dang bang state -> action."""
    state_names = state_names or {}
    action_names = action_names or {}

    print("state | action | y nghia")
    print("------+--------+------------------------")
    for state, action in enumerate(np.asarray(policy).astype(int)):
        state_label = state_names.get(state, f"state {state}")
        action_label = action_names.get(int(action), f"action {action}")
        print(f"{state:^6d}| {action:^7d}| {state_label} -> {action_label}")


def main() -> None:
    print("Deterministic policy:")
    print("policy =", policy)
    print("shape  =", policy.shape, "  (mot action cho moi state)")
    print()
    print_policy(policy, STATE_NAMES, ACTION_NAMES)

    print()
    print("Truy van action tai tung state:")
    for state in range(N_STATES):
        print(f"  policy[{state}] = {policy[state]} ({ACTION_NAMES[int(policy[state])]})")


if __name__ == "__main__":
    main()
