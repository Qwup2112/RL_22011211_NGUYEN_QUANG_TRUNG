"""Bai 16 - Thong tin co ban cua FrozenLake-v1 (4x4, is_slippery=True)."""

from mdp_utils import create_environment, env_sizes, get_transition_model


def main() -> None:
    env = create_environment(map_name="4x4", is_slippery=True)

    observation, info = env.reset(seed=42)
    n_states, n_actions = env_sizes(env)

    print("Number of states  :", n_states)
    print("Number of actions :", n_actions)
    print("Initial observation:", observation)
    print()
    print("Observation space :", env.observation_space)
    print("Action space      :", env.action_space)
    print("Info sau reset    :", info)
    print()

    desc = env.unwrapped.desc
    print("Ban do 4x4:")
    for row_index, row in enumerate(desc):
        cells = " ".join(cell.decode("ascii") for cell in row)
        print(f"  row {row_index}: {cells}")

    print()
    print("State index = row * 4 + col")
    print("Model chuyen trang thai co san tai env.unwrapped.P")
    transition_model = get_transition_model(env)
    print("So state trong model:", len(transition_model))
    print("So action cua state 0:", len(transition_model[0]))

    env.close()


if __name__ == "__main__":
    main()
