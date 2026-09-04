"""Bai 28 - So sanh FrozenLake tat dinh (is_slippery=False) va tron (True).

Moi truong hop chay 500 episode va so sanh success rate, average reward,
average episode length. Chuong trinh so sanh hai policy:
    - random policy   : chon action ngau nhien
    - planned policy   : di theo duong da tinh san tu Start toi Goal (Bai 26)
Policy thu hai cho thay ro anh huong cua tinh ngau nhien trong transition.
"""

import gymnasium as gym
import numpy as np

# Duong di toi uu tren ban do 4x4 mac dinh: 0 -> 4 -> 8 -> 9 -> 10 -> 14 -> 15
PLANNED_ACTIONS = {0: 1, 4: 1, 8: 2, 9: 2, 10: 1, 14: 2}
N_COLS = 4


def random_policy_factory(env: gym.Env):
    """Policy ngau nhien."""

    def policy(state):
        return env.action_space.sample()

    return policy


def planned_policy(state):
    """Policy tat dinh: di theo ke hoach, o cac o ngoai ke hoach thi di xuong/phai."""
    if state in PLANNED_ACTIONS:
        return PLANNED_ACTIONS[state]

    row, _ = divmod(int(state), N_COLS)
    return 1 if row < N_COLS - 1 else 2      # 1 = DOWN, 2 = RIGHT


def evaluate(is_slippery: bool, policy_name: str,
             n_episodes: int = 500, seed: int = 42) -> dict:
    """Chay mot policy tren FrozenLake va tra ve cac chi so tong hop."""
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    policy = random_policy_factory(env) if policy_name == "random" else planned_policy

    rewards = []
    lengths = []

    for _ in range(n_episodes):
        state, info = env.reset()
        total_reward = 0.0
        length = 0

        while True:
            action = policy(state)
            state, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            length += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(length)

    env.close()

    rewards_array = np.array(rewards)
    return {
        "policy": policy_name,
        "is_slippery": is_slippery,
        "success": int((rewards_array > 0).sum()),
        "success_rate": float((rewards_array > 0).mean()),
        "average_reward": float(rewards_array.mean()),
        "average_length": float(np.mean(lengths)),
    }


def main() -> None:
    n_episodes = 500

    results = [
        evaluate(False, "random", n_episodes),
        evaluate(True, "random", n_episodes),
        evaluate(False, "planned", n_episodes),
        evaluate(True, "planned", n_episodes),
    ]

    print(f"FrozenLake-v1, {n_episodes} episodes per setting")
    print()
    print("Policy  | is_slippery | Success | Success rate | Avg reward | Avg length")
    print("--------+-------------+---------+--------------+------------+-----------")
    for result in results:
        print(f"{result['policy']:<8s}| {str(result['is_slippery']):^12s}| "
              f"{result['success']:^8d}| {result['success_rate']:^13.4f}| "
              f"{result['average_reward']:^11.4f}| {result['average_length']:^10.2f}")

    random_det, random_slip, planned_det, planned_slip = results

    print()
    print("Random policy : deterministic "
          f"{random_det['success_rate']:.3f} vs slippery {random_slip['success_rate']:.3f}")
    print("Planned policy: deterministic "
          f"{planned_det['success_rate']:.3f} vs slippery {planned_slip['success_rate']:.3f}")

    # ------------------------------------------------------------------
    # KET LUAN (viet bang comment theo yeu cau cua de bai)
    #
    # 1. Voi is_slippery=False moi truong la TAT DINH: action nao thi agent di
    #    dung huong do. Mot chuoi action tinh san (planned policy) dat success
    #    rate 100% va episode chi dai dung 6 buoc.
    # 2. Voi is_slippery=True agent chi di dung huong voi xac suat 1/3, con 2/3
    #    bi truot sang hai huong vuong goc. Cung planned policy do, success rate
    #    tut xuong con khoang 2-6%: ke hoach tot khong con bao dam ket qua tot.
    # 3. Voi RANDOM policy thi hai che do gan nhu nhau (deu quanh 1-2%) va chenh
    #    lech giua chung chi la nhieu thong ke. Ly do: random policy von da di
    #    lang quang nen viec moi truong tron them cung khong lam no te hon may.
    #    Ket qua nay cho thay khong the danh gia do kho cua moi truong bang mot
    #    random policy - phai so sanh bang mot policy co chu dich.
    # 4. Episode o che do tron dai hon mot chut vi agent bi day di lung tung,
    #    trong khi o che do tat dinh no roi xuong ho rat nhanh.
    # 5. Ket luan chung: tinh ngau nhien cua transition lam bai toan kho hon han.
    #    Agent khong the chi "nho duong di", no phai hoc mot policy theo TRANG
    #    THAI, chiu duoc nhieu - day chinh la ly do can den cac thuat toan RL.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
