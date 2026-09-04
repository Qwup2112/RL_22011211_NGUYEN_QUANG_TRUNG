"""Bai 2 - Tao moi truong CartPole-v1, in doi tuong env va dong moi truong."""

import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1")

    # env la mot chuoi wrapper: TimeLimit -> OrderEnforcing -> PassiveEnvChecker
    # -> CartPoleEnv. In ra de thay ro cau truc nay.
    print("env object     :", env)
    print("type(env)      :", type(env))
    print("env.spec.id    :", env.spec.id)
    print("max_episode_steps:", env.spec.max_episode_steps)

    # env.unwrapped tra ve moi truong goc, khong con wrapper
    print("env.unwrapped  :", env.unwrapped)

    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()
