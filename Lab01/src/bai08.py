"""Bai 8 - Ham run_one_step(): thuc hien dung mot buoc tuong tac.

Ham tra ve day du observation, reward, terminated, truncated, info.
"""

import gymnasium as gym


def run_one_step(env: gym.Env, action: int):
    """Thuc hien mot action tren env va tra ve ket qua cua env.step()."""
    observation, reward, terminated, truncated, info = env.step(action)
    return observation, reward, terminated, truncated, info


def main() -> None:
    env = gym.make("CartPole-v1")

    observation, info = env.reset(seed=42)
    print("Initial observation:", observation)
    print()

    # Kiem thu voi 6 action (nhieu hon yeu cau toi thieu 5 action)
    test_actions = [0, 1, 1, 0, 1, 0]

    print("step | action | reward | terminated | truncated | observation")
    print("-----+--------+--------+------------+-----------+------------")
    for step_index, action in enumerate(test_actions):
        observation, reward, terminated, truncated, info = run_one_step(env, action)

        obs_text = " ".join(f"{value: .4f}" for value in observation)
        print(
            f"{step_index:^4d} | {action:^6d} | {reward:^6.1f} | "
            f"{str(terminated):^10s} | {str(truncated):^9s} | [{obs_text}]"
        )

        # Neu episode ket thuc som thi phai reset truoc khi step tiep
        if terminated or truncated:
            print("Episode ended -> reset environment.")
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
