"""Bai 5 - Mo phong Markov chain bang bo sinh so ngau nhien cua NumPy."""

import numpy as np

from bai01 import P, STATE_NAMES


def sample_next_state(current_state: int, P, rng: np.random.Generator) -> int:
    """Boc mot trang thai ke tiep theo hang P[current_state]."""
    probabilities = np.asarray(P, dtype=np.float64)[current_state]
    return int(rng.choice(len(probabilities), p=probabilities))


def simulate_chain(start_state: int, P, n_steps: int,
                   rng: np.random.Generator) -> list[int]:
    """Mo phong n_steps transition, tra ve chuoi state (ke ca state ban dau)."""
    states = [start_state]
    current_state = start_state

    for _ in range(n_steps):
        current_state = sample_next_state(current_state, P, rng)
        states.append(current_state)

    return states


def main() -> None:
    rng = np.random.default_rng(42)   # seed de tai lap duoc

    states = simulate_chain(0, P, 30, rng)

    print("Chuoi 30 transition bat dau tu Sunny:")
    print()
    print(" -> ".join(STATE_NAMES[state] for state in states))
    print()
    print("Chi so state:", states)

    counts = np.bincount(states, minlength=len(STATE_NAMES))
    print()
    print("Tan suat trong chuoi ngan nay:")
    for name, count in zip(STATE_NAMES, counts):
        print(f"  {name:<7s}: {count:2d}/{len(states)} = {count / len(states):.3f}")


if __name__ == "__main__":
    main()
