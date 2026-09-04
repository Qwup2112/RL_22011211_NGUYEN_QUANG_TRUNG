# Bài thực hành số 1 — Làm quen với Gymnasium

Họ tên: Nguyen Quang Trung

MSSV: 22011211

Lớp: K16_AI&RB

GitHub username: Qwup2112

Repository URL: <https://github.com/Qwup2112/RL_22011211_NGUYEN_QUANG_TRUNG>

Python version: 3.14.4

Gymnasium version: 1.3.0

NumPy version: 2.5.2

Matplotlib version: 3.11.1

Môi trường sử dụng: `CartPole-v1`, `FrozenLake-v1`, `MountainCar-v0`.
Toàn bộ code dùng **API Gymnasium mới** (`terminated` / `truncated`), không
dùng `import gym` và không dùng biến `done` của API cũ.

---

## Cách cài đặt

```bash
# 1. Tạo virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 2. Cài thư viện
cd Lab01
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Kiểm tra
python --version
pip show gymnasium
python src/bai01.py
```

---

## Cách chạy từng bài

Mọi script đều chạy độc lập từ thư mục `Lab01/`:

```bash
cd Lab01
python src/bai01.py
python src/bai02.py
...
python src/bai36.py
```

Riêng các bài `bai15`–`bai18`, `bai34`, `bai35` import lại hàm từ các bài
trước (`bai14`, `bai31`, `bai32`, `bai33`), nên phải chạy từ thư mục `Lab01`
(hoặc từ trong `src/`) để Python tìm thấy module.

### Bảng tra nhanh

| Phần | Bài | File | Nội dung |
|------|-----|------|----------|
| — | — | `src/starter.py` | Chương trình khởi động ở mục 6 của đề bài |
| **A** | 1 | `src/bai01.py` | In version Python / Gymnasium / NumPy / Matplotlib |
| A | 2 | `src/bai02.py` | Tạo `CartPole-v1`, in `env`, `env.close()` |
| A | 3 | `src/bai03.py` | Action space, đếm số action bằng code |
| A | 4 | `src/bai04.py` | Observation space: shape, dtype, low, high |
| A | 5 | `src/bai05.py` | `reset(seed=42)`, ý nghĩa từng phần tử observation |
| A | 6 | `src/bai06.py` | Sinh 20 action ngẫu nhiên + thống kê tần suất |
| **B** | 7 | `src/bai07.py` | Một bước tương tác đầy đủ |
| B | 8 | `src/bai08.py` | Hàm `run_one_step()` |
| B | 9 | `src/bai09.py` | Vòng lặp 20 timestep |
| B | 10 | `src/bai10.py` | Cộng dồn `total_reward` |
| B | 11 | `src/bai11.py` | Hàm `random_agent()` |
| B | 12 | `src/bai12.py` | Bỏ `done`, phân biệt Termination / Truncation |
| **C** | 13 | `src/bai13.py` | 10 episode, bảng Episode / Reward / Length |
| C | 14 | `src/bai14.py` | 100 episode, lưu `episode_rewards` ra `data/` |
| C | 15 | `src/bai15.py` | mean / min / max / std bằng NumPy |
| C | 16 | `src/bai16.py` | Episode tốt nhất (dùng `np.argmax`, không chạy lại env) |
| C | 17 | `src/bai17.py` | Biểu đồ `figures/reward_cartpole.png` |
| C | 18 | `src/bai18.py` | `moving_average()` + `figures/moving_average.png` |
| **D** | 19 | `src/bai19.py` | 10 env độc lập cùng `seed=42` |
| D | 20 | `src/bai20.py` | So sánh seed 42 và seed 100 |
| D | 21 | `src/bai21.py` | `env.action_space.seed()` |
| D | 22 | `src/bai22.py` | Hàm `experiment(seed, n_episodes)`, 6 seed |
| **E** | 23 | `src/bai23.py` | Tạo `FrozenLake-v1`, số state / số action |
| E | 24 | `src/bai24.py` | `render_mode="ansi"` |
| E | 25 | `src/bai25.py` | `ACTION_NAMES`, xác định bằng thí nghiệm |
| E | 26 | `src/bai26.py` | Chuỗi action đi từ Start tới Goal |
| E | 27 | `src/bai27.py` | 200 episode, `success_rate` |
| E | 28 | `src/bai28.py` | So sánh `is_slippery=False/True`, 500 episode |
| **F** | 29 | `src/bai29.py` | Policy dưới dạng hàm |
| F | 30 | `src/bai30.py` | `always_left_policy` / `always_right_policy` |
| F | 31 | `src/bai31.py` | `angle_based_policy` |
| F | 32 | `src/bai32.py` | Policy cải tiến (góc + vận tốc góc) |
| **G** | 33 | `src/bai33.py` | `run_episode()` tổng quát |
| G | 34 | `src/bai34.py` | `evaluate_policy()` |
| G | 35 | `src/bai35.py` | So sánh 3 agent + `figures/comparison_agents.png` |
| G | 36 | `src/bai36.py` | Mini-project hoàn chỉnh |
| Mục 8 | — | `src/migration_gym_to_gymnasium.py` | Chuyển code Gym cũ sang Gymnasium |
| Mục 9 | — | `src/main.py` | Chương trình tổng hợp |

---

## Cách chạy chương trình tổng hợp

```bash
cd Lab01

# Chương trình tổng hợp (khung ở mục 9 của đề bài)
python src/main.py

# Mini-project đầy đủ của Bài 36
python src/bai36.py

# Notebook
jupyter notebook notebooks/Lab01_22011211_NguyenQuangTrung.ipynb
```

Chạy tất cả các bài một lượt (Git Bash / Linux / macOS):

```bash
cd Lab01
for f in src/bai*.py; do echo "=== $f ==="; python "$f"; done
```

PowerShell:

```powershell
cd Lab01
Get-ChildItem src\bai*.py | ForEach-Object { Write-Host "=== $_ ==="; python $_.FullName }
```

---

## Mô tả kết quả

### CartPole-v1 — so sánh 3 agent (Bài 35, 500 episode mỗi agent, seed = 42)

| Agent | Mean reward | Std | Min | Max | Mean length |
|-------|-------------|-----|-----|-----|-------------|
| Random | 21.59 | 10.42 | 8.0 | 73.0 | 21.59 |
| Angle-based | 42.21 | 9.03 | 24.0 | 72.0 | 42.21 |
| Improved (góc + vận tốc góc) | 500.00 | 0.00 | 500.0 | 500.0 | 500.00 |

- Random agent chỉ trụ được ~21 bước vì không dùng chút thông tin nào từ
  observation; 500/500 episode kết thúc bằng `terminated=True` (cột ngã).
- Angle-based policy dùng `observation[2]` (góc của pole) đã tốt gấp đôi.
- Improved policy dùng thêm `observation[3]` (vận tốc góc) để **dự đoán** góc
  sắp tới nên phản ứng sớm hơn: 500/500 episode kết thúc bằng
  `truncated=True` — chạm giới hạn 500 bước, tức kết quả tốt nhất có thể của
  `CartPole-v1`. Std bằng 0 cho thấy luật này ổn định với mọi trạng thái khởi
  tạo ngẫu nhiên, không phải ăn may.

### FrozenLake-v1 — deterministic vs stochastic (Bài 28, 500 episode mỗi cấu hình)

| Policy | `is_slippery` | Success rate | Avg reward | Avg length |
|--------|---------------|--------------|------------|------------|
| random | False | 0.0140 | 0.0140 | 7.70 |
| random | True | 0.0200 | 0.0200 | 7.99 |
| planned | False | **1.0000** | 1.0000 | 6.00 |
| planned | True | **0.0320** | 0.0320 | 5.06 |

- Với `is_slippery=False`, môi trường tất định nên một chuỗi action tính sẵn
  (Bài 26) thắng 100% và chỉ mất đúng 6 bước.
- Với `is_slippery=True`, agent chỉ đi đúng hướng với xác suất 1/3 nên cùng
  chuỗi action đó tụt xuống ~3%. Đây là bằng chứng rõ nhất cho việc **tính ngẫu
  nhiên trong transition làm bài toán khó hơn hẳn**.
- Random policy gần như không phân biệt được hai cấu hình (1.4% vs 2.0%, chỉ là
  nhiễu thống kê) — không thể dùng random policy để đánh giá độ khó của một môi
  trường.

### Biểu đồ

| File | Sinh ra bởi | Nội dung |
|------|-------------|----------|
| `figures/reward_cartpole.png` | `bai17.py` | Reward theo 100 episode của random agent |
| `figures/moving_average.png` | `bai18.py` | Reward + moving average (window = 10) |
| `figures/comparison_agents.png` | `bai35.py` | So sánh mean reward của 3 agent |
| `figures/mini_project_rewards.png` | `bai36.py` | Reward và episode length của mini-project |
| `figures/mini_project_moving_average.png` | `bai36.py` | Moving average của mini-project |
| `figures/main_summary.png` | `main.py` | Tổng hợp random vs heuristic |

---

## Khó khăn gặp phải

1. **Phân biệt `terminated` và `truncated`.** Ban đầu rất dễ gộp lại thành một
   biến `done` như API Gym cũ. Bài 12 và `migration_gym_to_gymnasium.py` giải
   thích tại sao không được làm vậy: khi học giá trị, `terminated` phải cho
   target `= reward`, còn `truncated` vẫn phải bootstrap
   `= reward + gamma * V(next_state)`.
2. **Seed ở hai chỗ khác nhau.** `env.reset(seed=...)` chỉ seed bộ sinh số của
   *môi trường*; chuỗi action ngẫu nhiên lại do `env.action_space` sinh ra nên
   phải seed riêng bằng `env.action_space.seed(...)` (Bài 21). Thiếu một trong
   hai thì thí nghiệm không tái lập được.
3. **Seed lại ở mỗi episode làm hỏng thí nghiệm.** Nếu gọi `reset(seed=42)` ở
   *mọi* episode thì cả 100 episode đều giống hệt nhau. Cách đúng: seed một lần
   ở episode đầu, các episode sau gọi `reset()` không tham số để tiếp tục chuỗi
   ngẫu nhiên đó.
4. **`Taxi-v3` đã bị deprecate** trong Gymnasium 1.3.0 (`gym.make` báo lỗi
   `DeprecatedEnv`, yêu cầu `Taxi-v4`). Bài 33 dùng `MountainCar-v0` thay thế —
   môi trường này còn cho thấy rõ trường hợp `truncated=True`.
5. **Backend của Matplotlib.** Chạy script từ terminal mà gọi `plt.show()` sẽ
   treo chương trình. Các file có vẽ hình đều đặt `matplotlib.use("Agg")` trước
   khi `import matplotlib.pyplot` rồi lưu thẳng ra `figures/`.
6. **Xác định ý nghĩa action của FrozenLake.** Thay vì tin vào tài liệu, Bài 25
   đặt agent vào ô giữa bản đồ (`env.unwrapped.s = 9`), thử cả 4 action và suy
   ra hướng di chuyển từ chênh lệch chỉ số state — kết quả xác nhận
   `0=LEFT, 1=DOWN, 2=RIGHT, 3=UP`.

---

## Kết luận

- Vòng lặp tương tác `reset() → policy(observation) → step(action) →
  terminated/truncated` là khung xương của mọi thuật toán RL sau này; toàn bộ 36
  bài đều xoay quanh đúng vòng lặp đó.
- **Không có agent nào trong bài này học cả.** Random, always-left, angle-based
  và improved đều là luật cố định, không có tham số nào được cập nhật từ reward.
  Vì vậy đường moving average nằm ngang — đây chính là dấu hiệu để phân biệt với
  một thuật toán RL thật (Q-Learning, SARSA, DQN), nơi đường này phải đi lên.
- Chỉ cần *dùng thêm thông tin từ observation* là hiệu quả đã tăng từ 21.6 lên
  500.0 reward. Điều mà RL làm là **tự tìm ra** luật đó thay vì để con người
  thiết kế tay.
- Phải chạy nhiều episode và có seed: một episode đơn lẻ của random agent có thể
  đạt 73 điểm (cao gấp 3 lần trung bình) và sẽ dẫn tới kết luận sai hoàn toàn.
- Reward trung bình là thước đo chính, nhưng phải đọc kèm std: hai agent cùng
  mean khác std là hai agent rất khác nhau về độ ổn định.

---

## Tài liệu tham khảo

- Tài liệu chính thức Gymnasium: <https://gymnasium.farama.org/>
- Mô tả môi trường CartPole: <https://gymnasium.farama.org/environments/classic_control/cart_pole/>
- Mô tả môi trường FrozenLake: <https://gymnasium.farama.org/environments/toy_text/frozen_lake/>
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd edition.

Toàn bộ code trong thư mục này do sinh viên tự viết; các nguồn trên chỉ dùng để
tra cứu ý nghĩa observation, action và tham số của môi trường.
