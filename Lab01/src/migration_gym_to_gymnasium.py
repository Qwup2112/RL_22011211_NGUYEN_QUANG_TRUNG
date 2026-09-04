"""Phan bat buoc (muc 8) - Chuyen code Gym cu sang Gymnasium hien tai.

CODE GYM CU (khong duoc dung nua):

    import gym

    env = gym.make("CartPole-v0")
    observation = env.reset()

    for t in range(1000):
        env.render()

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)

        if done:
            break

Nhung diem phai sua khi chuyen sang Gymnasium:

    1. import gym                -> import gymnasium as gym
    2. env.reset()               -> observation, info = env.reset(seed=...)
       (API moi tra ve MOT TUPLE gom observation va info)
    3. env.step() tra ve 4 gia tri (obs, reward, done, info)
       -> tra ve 5 gia tri (obs, reward, terminated, truncated, info)
    4. env.render() goi trong vong lap -> render_mode duoc khai bao ngay
       tu gym.make(..., render_mode="human"); goi env.render() khong tham so
       trong vong lap khong con dung cach.
    5. Phai goi env.close() khi ket thuc.
    6. CartPole-v0 da cu (gioi han 200 buoc) -> dung CartPole-v1 (500 buoc).
"""

import gymnasium as gym

# ---------------------------------------------------------------------------
# GIAI THICH BAT BUOC
#
# terminated co y nghia gi?
#   terminated = True khi episode ket thuc theo DUNG BAN CHAT cua bai toan:
#   agent di vao mot trang thai ket thuc (terminal state). Voi CartPole la khi
#   cot nghieng qua 12 do hoac xe chay ra khoi ray; voi FrozenLake la khi agent
#   roi xuong ho hoac toi duoc Goal. Sau trang thai nay khong con tuong lai
#   nao de tinh tiep, gia tri cua trang thai ke tiep bang 0.
#
# truncated co y nghia gi?
#   truncated = True khi episode bi CAT NGANG boi mot gioi han BEN NGOAI bai
#   toan, thuong la gioi han so buoc toi da (wrapper TimeLimit, CartPole-v1 la
#   500 buoc). Ban than bai toan chua ket thuc, agent van con the tiep tuc neu
#   duoc phep. Vi vay trang thai cuoi cung o day VAN co gia tri tuong lai.
#
# Vi sao khong nen dung done cua API cu?
#   API cu gop hai truong hop tren vao mot bien done duy nhat nen agent khong
#   phan biet duoc "ket thuc that su" voi "het gio". Khi hoc gia tri (Q-Learning,
#   DQN...) hai truong hop nay phai duoc xu ly KHAC NHAU:
#       - terminated: target = reward
#       - truncated : target = reward + gamma * V(next_state)  (bootstrapping)
#   Dung done chung se bootstrap sai o cac episode bi cat ngang, lam sai lech
#   gia tri hoc duoc. Do do Gymnasium tach done thanh terminated va truncated.
# ---------------------------------------------------------------------------


def run_migrated_program(n_steps: int = 1000, seed: int = 42) -> None:
    """Ban viet lai cua chuong trinh Gym cu theo API Gymnasium hien tai."""
    # 1. import gymnasium as gym (o tren) va dung CartPole-v1
    env = gym.make("CartPole-v1")

    # 2. reset() tra ve tuple (observation, info)
    observation, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    total_reward = 0.0

    for t in range(n_steps):
        # 3. Khong goi env.render() trong vong lap. Neu muon xem hinh anh thi
        #    khai bao ngay khi tao moi truong:
        #        env = gym.make("CartPole-v1", render_mode="human")

        action = env.action_space.sample()

        # 4. step() tra ve 5 gia tri, khong con bien done
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)

        # 5. Tu tao dieu kien dung tu terminated va truncated
        if terminated or truncated:
            reason = "Termination" if terminated else "Truncation"
            print(f"Episode ended at step t={t} -> {reason}")
            print(f"  terminated = {terminated}")
            print(f"  truncated  = {truncated}")
            break
    else:
        print(f"Reached the loop limit of {n_steps} steps.")

    print(f"Total reward: {total_reward:.1f}")

    # 6. Luon dong moi truong
    env.close()


def main() -> None:
    print("Old Gym program rewritten with the current Gymnasium API")
    print("-" * 56)
    run_migrated_program()


if __name__ == "__main__":
    main()
