"""Bai 5 - Quan sat trang thai ban dau cua CartPole sau reset(seed=42)."""

import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1")

    observation, info = env.reset(seed=42)

    print("Observation:", observation)
    print("Type       :", type(observation))
    print("Shape      :", observation.shape)
    print("Dtype      :", observation.dtype)
    print("Info       :", info)

    # Y nghia va kieu du lieu cua tung phan tu trong observation:
    # observation[0] = cart position        -> numpy.float32 (so thuc, met)
    # observation[1] = cart velocity        -> numpy.float32 (so thuc, m/s)
    # observation[2] = pole angle           -> numpy.float32 (so thuc, radian)
    # observation[3] = pole angular velocity-> numpy.float32 (so thuc, rad/s)
    # Ca 4 phan tu deu la so thuc lien tuc, gom trong mot numpy.ndarray float32.
    labels = [
        "cart position",
        "cart velocity",
        "pole angle (rad)",
        "pole angular velocity",
    ]
    print()
    for index, label in enumerate(labels):
        value = observation[index]
        print(f"observation[{index}] = {value: .6f}  ({label}, {type(value).__name__})")

    # info cua CartPole sau reset la dictionary rong: moi truong khong cung cap
    # them thong tin phu (khac voi FrozenLake co info['prob']).
    print()
    print("info is empty dict:", info == {})

    env.close()


if __name__ == "__main__":
    main()
