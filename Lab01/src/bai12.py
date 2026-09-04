"""Bai 12 - Random agent khong dung bien done cua API Gym cu.

Chi nhan terminated va truncated tu env.step(), tu tao bien episode_finished
va in ro nguyen nhan ket thuc episode.
"""

import gymnasium as gym


def random_agent_no_done(env: gym.Env, max_steps: int = 500):
    """Chay mot episode va tra ve (total_reward, length, terminated, truncated)."""
    observation, info = env.reset()

    total_reward = 0.0
    episode_length = 0
    terminated = False
    truncated = False

    while episode_length < max_steps:
        action = env.action_space.sample()

        # KHONG viet: observation, reward, done, info = env.step(action)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        episode_length += 1

        # Bien "done" duoc tu tao o phia agent, khong lay tu moi truong
        episode_finished = terminated or truncated
        if episode_finished:
            break

    return total_reward, episode_length, terminated, truncated


def describe_ending(terminated: bool, truncated: bool) -> str:
    """Cho biet nguyen nhan episode ket thuc."""
    if terminated:
        # terminated: pole nga qua 12 do hoac cart ra khoi ray -> ket thuc that su
        return "Termination"
    if truncated:
        # truncated: cham gioi han 500 buoc cua wrapper TimeLimit -> bi cat ngang
        return "Truncation"
    return "Still running (reached the max_steps of the agent)"


def main() -> None:
    env = gym.make("CartPole-v1")

    env.reset(seed=42)
    env.action_space.seed(42)

    for episode in range(5):
        total_reward, length, terminated, truncated = random_agent_no_done(env)

        print(f"Episode {episode + 1}: reward={total_reward:6.1f}, length={length:3d}, "
              f"terminated={terminated}, truncated={truncated}")
        print(f"  Reason: {describe_ending(terminated, truncated)}")

    env.close()


if __name__ == "__main__":
    main()
