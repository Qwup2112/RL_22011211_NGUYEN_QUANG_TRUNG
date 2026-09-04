"""Bai 33 - Ham run_episode() tong quat.

Ham chay dung mot episode cho BAT KY moi truong Gymnasium nao (khong phu thuoc
rieng CartPole) va tra ve mot dictionary mo ta ket qua.
"""

import gymnasium as gym


def run_episode(env: gym.Env, policy, seed=None, max_steps: int = 1000) -> dict:
    """Chay mot episode voi policy cho truoc.

    Tham so:
        env       : moi truong Gymnasium bat ky.
        policy    : ham policy(observation) tra ve mot action hop le.
        seed      : neu khac None thi truyen vao env.reset(seed=seed).
        max_steps : gioi han so buoc do agent tu dat.

    Tra ve dictionary: reward, length, terminated, truncated.
    """
    if seed is None:
        observation, info = env.reset()
    else:
        observation, info = env.reset(seed=seed)

    total_reward = 0.0
    length = 0
    terminated = False
    truncated = False

    while length < max_steps:
        action = policy(observation)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        length += 1

        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "length": length,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def make_random_policy(env_name: str, seed: int = 0):
    """Tao policy ngau nhien dung cho mot ten moi truong bat ky.

    Action space duoc lay tu mot env tam thoi roi seed lai, nen policy nay
    dung duoc voi moi moi truong ma khong phu thuoc CartPole.
    """
    probe_env = gym.make(env_name)
    action_space = probe_env.action_space
    action_space.seed(seed)
    probe_env.close()

    def policy(observation):
        return action_space.sample()

    return policy


def main() -> None:
    # Chung minh ham tong quat: dung cung run_episode cho 3 moi truong khac nhau
    env_names = ["CartPole-v1", "FrozenLake-v1", "MountainCar-v0"]

    print("Env name      | Reward | Length | Terminated | Truncated")
    print("--------------+--------+--------+------------+----------")

    for env_name in env_names:
        env = gym.make(env_name)
        policy = make_random_policy(env_name, seed=42)

        result = run_episode(env, policy, seed=42, max_steps=1000)

        print(f"{env_name:<14s}| {result['reward']:^7.1f}| {result['length']:^7d}| "
              f"{str(result['terminated']):^11s}| {str(result['truncated']):^9s}")

        env.close()

    print()
    print("Full dictionary of the last episode:")
    print(result)


if __name__ == "__main__":
    main()
