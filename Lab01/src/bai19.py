"""Bai 19 - Thu nghiem voi seed: reset(seed=42) tren 10 environment doc lap."""

import gymnasium as gym
import numpy as np


def collect_initial_observations(env_name: str = "CartPole-v1",
                                 seed: int = 42,
                                 n_environments: int = 10) -> list[np.ndarray]:
    """Tao n_environment doc lap, reset moi cai voi cung mot seed."""
    observations = []

    for _ in range(n_environments):
        env = gym.make(env_name)                 # moi vong lap la mot env moi
        observation, info = env.reset(seed=seed)
        observations.append(np.array(observation, copy=True))
        env.close()                              # dong ngay sau khi dung xong

    return observations


def main() -> None:
    seed = 42
    observations = collect_initial_observations(seed=seed, n_environments=10)

    print(f"Initial observation after reset(seed={seed}) in 10 independent envs:")
    print()
    for index, observation in enumerate(observations, start=1):
        values = " ".join(f"{value: .8f}" for value in observation)
        print(f"env {index:2d}: [{values}]")

    reference = observations[0]
    all_equal = all(np.array_equal(reference, observation) for observation in observations)

    print()
    print("All observations identical:", all_equal)

    # Ket luan:
    # Ca 10 environment doc lap deu tra ve cung mot initial observation khi
    # duoc reset voi cung seed=42, vi seed quyet dinh hoan toan bo sinh so
    # ngau nhien cua moi truong. Nho vay thi nghiem RL co the tai lap chinh xac.
    print()
    print("Conclusion: the same seed always produces the same initial state,")
    print("so seeding makes the experiment reproducible.")


if __name__ == "__main__":
    main()
