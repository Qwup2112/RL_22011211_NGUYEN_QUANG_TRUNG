# Lab02 - Markov Decision Process và Dynamic Programming

## Thông tin sinh viên

- Họ tên: Nguyen Quang Trung
- MSSV: 22011211
- Lớp: K16_AI&RB
- GitHub username: Qwup2112
- Repository: <https://github.com/Qwup2112/RL_22011211_NGUYEN_QUANG_TRUNG>

Phiên bản đang dùng: Python 3.14.4 · Gymnasium 1.3.0 · NumPy 2.5.2 · Matplotlib 3.11.1

## Mục tiêu

Chuyển từ việc *tương tác* với môi trường (Lab01) sang việc **mô hình hóa và
giải** bài toán bằng Markov Decision Process:

- biểu diễn Markov chain bằng transition matrix, kiểm tra tính hợp lệ, mô phỏng;
- tính return, phân tích ảnh hưởng của discount factor `gamma`;
- biểu diễn một MDP rời rạc, deterministic policy và stochastic policy;
- đọc model chuyển trạng thái `env.unwrapped.P` của `FrozenLake-v1`;
- tự cài đặt Bellman backup, Policy Evaluation, Policy Improvement,
  Policy Iteration và Value Iteration;
- trích xuất optimal policy, đánh giá bằng simulation và so sánh hai thuật toán.

Toàn bộ thuật toán Dynamic Programming đều **tự lập trình bằng NumPy**; không
gọi bất kỳ thư viện RL nào có sẵn `value_iteration()` / `policy_iteration()`,
và không hard-code optimal policy.

## Cấu trúc thư mục

```text
Lab02/
├── README.md
├── requirements.txt
├── src/
│   ├── bai01.py … bai36.py
│   ├── mdp_utils.py          # thư viện dùng chung (mục 14 của đề)
│   └── main.py               # chương trình tổng hợp (mục 11 của đề)
├── notebooks/
│   └── Lab02_22011211_NguyenQuangTrung.ipynb
├── figures/
│   ├── markov_distribution.png
│   ├── gamma_comparison.png
│   ├── value_iteration_convergence.png
│   ├── policy_iteration_convergence.png
│   ├── algorithm_comparison.png
│   └── … (biểu đồ phụ)
└── data/
    ├── README.md
    └── value_iteration_deltas.csv
```

### Về `mdp_utils.py`

Theo mục 14 của đề, mọi hàm dùng chung được cài đặt **một lần duy nhất** trong
`src/mdp_utils.py`; các file `baiXX.py` import lại thay vì copy-paste:

`create_environment`, `get_transition_model`, `q_from_v`, `action_values`,
`policy_evaluation_sweep`, `policy_evaluation`, `greedy_policy_from_value`,
`policy_iteration`, `value_iteration_sweep`, `value_iteration`,
`evaluate_policy_by_simulation`, `print_value_grid`, `print_policy_grid`.

## Cài đặt

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

cd Lab02
pip install -r requirements.txt
```

## Cách chạy

```bash
cd Lab02
python src/bai01.py
python src/bai24.py      # Iterative Policy Evaluation
python src/bai29.py      # Policy Iteration
python src/bai32.py      # Value Iteration
python src/bai35.py      # So sánh hai thuật toán
python src/bai36.py      # Mini-project đầy đủ
python src/main.py       # Chương trình tổng hợp
```

`main.py` nhận tham số dòng lệnh:

```bash
python src/main.py --gamma 0.9 --theta 1e-6
python src/main.py --map-name 8x8 --no-slippery
python src/main.py --all          # chạy cả is_slippery=False và True
```

Chạy toàn bộ (Git Bash):

```bash
cd Lab02
for f in src/bai*.py; do echo "=== $f ==="; python "$f"; done
```

Notebook:

```bash
jupyter notebook notebooks/Lab02_22011211_NguyenQuangTrung.ipynb
```

### Bảng tra nhanh 36 bài

| Phần | Bài | File | Nội dung |
|------|-----|------|----------|
| **A** | 1 | `src/bai01.py` | Transition matrix 3×3 (Sunny/Cloudy/Rainy) |
| A | 2 | `src/bai02.py` | `validate_transition_matrix()` |
| A | 3 | `src/bai03.py` | Phân phối sau một bước `p1 = p0 @ P` |
| A | 4 | `src/bai04.py` | `state_distribution()` + `markov_distribution.png` |
| A | 5 | `src/bai05.py` | `sample_next_state()`, mô phỏng 30 transition |
| A | 6 | `src/bai06.py` | 100 000 transition: lý thuyết vs mô phỏng |
| **B** | 7 | `src/bai07.py` | `compute_return()`, `gamma = 1.0` |
| B | 8 | `src/bai08.py` | Bảng return theo 5 giá trị gamma |
| B | 9 | `src/bai09.py` | `discounted_returns()` tính ngược từ cuối episode |
| B | 10 | `src/bai10.py` | `gamma_comparison.png` |
| B | 11 | `src/bai11.py` | Reward sớm vs trễ, tìm `gamma*` |
| **C** | 12 | `src/bai12.py` | MDP 2 state × 2 action |
| C | 13 | `src/bai13.py` | `validate_mdp()` |
| C | 14 | `src/bai14.py` | Deterministic policy + `print_policy()` |
| C | 15 | `src/bai15.py` | Stochastic policy, kiểm tra tổng xác suất |
| **D** | 16 | `src/bai16.py` | Thông tin cơ bản của FrozenLake |
| D | 17 | `src/bai17.py` | In transition model của state 0 |
| D | 18 | `src/bai18.py` | `describe_state()` cho state 0, 1, 14 |
| D | 19 | `src/bai19.py` | Kiểm tra tổng xác suất mọi (s, a) |
| D | 20 | `src/bai20.py` | Deterministic vs stochastic tại (0, RIGHT) |
| **E** | 21 | `src/bai21.py` | `q_from_v()` + kiểm chứng từng số hạng |
| E | 22 | `src/bai22.py` | `action_values()` |
| E | 23 | `src/bai23.py` | `policy_evaluation_sweep()` |
| E | 24 | `src/bai24.py` | `policy_evaluation()` |
| E | 25 | `src/bai25.py` | Theo dõi hội tụ + `policy_evaluation_convergence.png` |
| **F** | 26 | `src/bai26.py` | `greedy_policy_from_value()` |
| F | 27 | `src/bai27.py` | `print_frozenlake_policy()` với mũi tên |
| F | 28 | `src/bai28.py` | Một bước Policy Improvement, đếm state đổi action |
| F | 29 | `src/bai29.py` | `policy_iteration()` + `policy_iteration_convergence.png` |
| F | 30 | `src/bai30.py` | Kiểm tra `policy_stable` |
| **G** | 31 | `src/bai31.py` | `value_iteration_sweep()` |
| G | 32 | `src/bai32.py` | `value_iteration()` + `value_iteration_convergence.png` |
| G | 33 | `src/bai33.py` | Trích xuất optimal policy |
| **H** | 34 | `src/bai34.py` | `evaluate_policy_by_simulation()` |
| H | 35 | `src/bai35.py` | So sánh VI/PI + `algorithm_comparison.png` |
| H | 36 | `src/bai36.py` | Mini-project DP Solver |

## Thuật toán đã cài đặt

### Bellman backup

```text
Q(s,a) = sum_(s') p(s'|s,a) * [ r + gamma * V(s') ]
```

Cài đặt trong `q_from_v()`. Hai điểm quan trọng:

- **Duyệt mọi transition** của `(state, action)` rồi cộng theo xác suất. Với
  `is_slippery=True` mỗi cặp `(s, a)` có 3 transition (xác suất 1/3 mỗi cái);
  chỉ lấy transition đầu tiên sẽ cho value function sai hoàn toàn.
- **Xử lý terminal transition:** nếu `terminated=True` thì *không* bootstrap
  tiếp (`future_value = 0`), vì trạng thái kết thúc không còn tương lai:

  ```python
  future_value = 0.0 if terminated else gamma * V[next_state]
  q_value += probability * (reward + future_value)
  ```

### Policy Evaluation

```text
V_pi(s) <- sum_a pi(a|s) * sum_(s') p(s'|s,a) * [ r + gamma * V_pi(s') ]
```

Lặp cho tới khi `delta = max|V_mới − V_cũ| < theta`. Chấp nhận cả policy
deterministic (mảng 1 chiều) lẫn stochastic (ma trận `n_states × n_actions`).

Kết quả với uniform random policy, `gamma = 0.99`, `theta = 1e-8`:
hội tụ sau **71 iteration**, `V(state 0) = 0.012356`.

### Policy Iteration

Khởi tạo policy → Policy Evaluation → Policy Improvement → kiểm tra
`policy_stable` → lặp hoặc dừng. Biến `policy_stable` được tự tính bằng
`np.all(new_policy == policy)`.

Kết quả: hội tụ sau **7 vòng**, tổng **1 426** Bellman sweep.

### Value Iteration

```text
V*(s) <- max_a sum_(s') p(s'|s,a) * [ r + gamma * V*(s') ]
```

Kết quả: hội tụ sau **438 iteration** (`gamma = 0.99`, `theta = 1e-8`).
Số iteration phụ thuộc mạnh vào gamma: 112 (γ=0.90) → 184 (γ=0.95) → 438 (γ=0.99).

## Kết quả FrozenLake

### Optimal state values `V*` (4×4, `is_slippery=True`, `gamma = 0.99`)

```text
0.5420  0.4988  0.4707  0.4569
0.5585  0.0000  0.3583  0.0000
0.5918  0.6431  0.6152  0.0000
0.0000  0.7417  0.8628  0.0000
```

### Optimal policy

```text
←  ↑  ↑  ↑
←  H  ←  H
↑  ↓  ←  H
H  →  ↓  G
```

Policy tối ưu **không đi đường ngắn nhất**. Ví dụ ở state 0 nó chọn `LEFT` —
đâm vào tường. Đó lại là lựa chọn an toàn nhất: khi trượt, hai hướng vuông góc
với `LEFT` là `UP` (tường) và `DOWN` (state 4), nên agent **không thể** rơi vào
hố ở state 5. Đây là điểm rất dễ nhầm nếu chỉ nhìn bản đồ.

### Đánh giá bằng simulation (1000 episode, seed = 42)

| Policy | Success | Success rate | Mean reward | Mean length | Min | Max |
|--------|--------:|-------------:|------------:|------------:|----:|----:|
| Random (cố định, ngẫu nhiên) | 27 | 0.0270 | 0.0270 | 7.29 | 2 | 35 |
| Value Iteration | 724 | 0.7240 | 0.7240 | 44.03 | 6 | 100 |
| Policy Iteration | 724 | 0.7240 | 0.7240 | 44.03 | 6 | 100 |

`V*(state 0) = 0.5420` nhưng success rate đo được là `0.7240` — hai con số này
**không phải cùng một đại lượng**: `V*` là return *có chiết khấu* (γ = 0.99, và
episode dài trung bình 44 bước nên hệ số chiết khấu đáng kể), còn success rate
là tỷ lệ thắng không chiết khấu.

### Deterministic vs stochastic (mini-project, Bài 36)

| Cấu hình | Thuật toán | Vòng lặp | Bellman sweep | Thời gian | Success rate |
|----------|------------|---------:|--------------:|----------:|-------------:|
| `is_slippery=False` | Value Iteration | 7 | 7 | 0.8 ms | 1.000 |
| `is_slippery=False` | Policy Iteration | 7 | 29 | 3.0 ms | 1.000 |
| `is_slippery=True` | Value Iteration | 438 | 438 | 51.2 ms | 0.724 |
| `is_slippery=True` | Policy Iteration | 7 | 1 426 | 154.0 ms | 0.724 |

## So sánh Value Iteration và Policy Iteration

| Thuật toán | Số vòng lặp | Bellman sweep | Thời gian | Success rate | Mean reward |
|---|---:|---:|---:|---:|---:|
| Value Iteration | 438 | 438 | 50.3 ms | 0.7240 | 0.7240 |
| Policy Iteration | 7 | 1 426 | 148.0 ms | 0.7240 | 0.7240 |

Biểu đồ: `figures/algorithm_comparison.png`.

## Nhận xét

1. Hai thuật toán cho ra **cùng một optimal policy** và cùng `V*` (sai lệch
   `1.3e-09`, đúng bằng mức của `theta`) — đúng như lý thuyết: cả hai đều hội tụ
   về nghiệm duy nhất của Bellman optimality equation.
2. Policy Iteration chỉ cần **7 vòng chính** so với 438 iteration của Value
   Iteration, vì mỗi vòng nó giải gần như chính xác `V_pi`.
3. Nhưng đếm theo **tổng số Bellman sweep** thì Policy Iteration tốn hơn hẳn
   (1 426 so với 438) vì mỗi vòng phải chạy Policy Evaluation đến hội tụ.
4. Kết quả là trên bản đồ 4×4 này **Value Iteration nhanh hơn khoảng 3 lần**.
   Số vòng lặp chính ít không có nghĩa là rẻ hơn — chi phí thật nằm ở tổng số
   Bellman backup.
5. Policy Iteration có lợi thế khi không gian action lớn, hoặc khi Policy
   Evaluation được giải bằng hệ phương trình tuyến tính thay vì lặp.
6. Value Iteration đơn giản hơn về cài đặt: chỉ một vòng lặp, không cần tách
   hai pha evaluation/improvement.
7. Success rate 0.724 là **giới hạn của chính bài toán**, không phải lỗi thuật
   toán: môi trường trượt khiến ~27% số episode kết thúc dưới hố dù agent chơi
   tối ưu. Với `is_slippery=False` thì DP đạt 100%.
8. Cả hai đều là **model-based**: bắt buộc phải truy cập `env.unwrapped.P`.
   Không biết transition model thì không dùng trực tiếp được — phải chuyển sang
   Monte Carlo hoặc Temporal-Difference Learning ở các Lab sau.
9. `gamma` ảnh hưởng mạnh đến chi phí: γ càng gần 1, thông tin phải lan truyền
   càng xa nên cả hai thuật toán đều cần nhiều sweep hơn.
10. `theta` đánh đổi giữa độ chính xác và số iteration: `1e-2` chỉ cần 7 vòng
    Policy Evaluation nhưng `V(0)` sai tới 87%; `1e-8` cần 71 vòng và cho kết
    quả chính xác tới 8 chữ số.

## Khó khăn gặp phải

1. **Terminal transition.** Nếu bootstrap cả ở transition có `terminated=True`
   thì `V` của các ô hố/Goal bị "rò rỉ" giá trị ngược lại và policy sai. Phải
   nhân thêm hệ số `(1 − terminated)` như trong `q_from_v()`.
2. **Hai dạng policy.** Bài 14/26/34 dùng deterministic policy (mảng chỉ số
   action) còn Bài 15/23/24 dùng stochastic policy (ma trận xác suất).
   `policy_evaluation()` phải nhận được cả hai, nên có `as_stochastic_policy()`
   để chuẩn hoá về one-hot.
3. **Ký tự mũi tên `← ↓ → ↑`.** Trên Windows, `sys.stdout` khi bị redirect dùng
   cp1252 và ném `UnicodeEncodeError`. `mdp_utils.py` gọi
   `sys.stdout.reconfigure(encoding="utf-8")` và vẫn giữ bảng ASCII dự phòng
   `< v > ^` nếu terminal không hiển thị được.
4. **Đọc nhầm `V*` thành xác suất thắng.** `V*(0) = 0.54` nhưng success rate là
   `0.72` — phải nhớ `V*` là return *có chiết khấu*.

## Tài liệu tham khảo

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., chương 3
  (Finite MDPs) và chương 4 (Dynamic Programming).
- Tài liệu Gymnasium: <https://gymnasium.farama.org/>
- FrozenLake: <https://gymnasium.farama.org/environments/toy_text/frozen_lake/>

Toàn bộ code trong thư mục này do sinh viên tự viết; các nguồn trên chỉ dùng để
tra cứu định dạng transition model và ý nghĩa tham số của môi trường.
