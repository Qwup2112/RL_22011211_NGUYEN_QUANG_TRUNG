# BÀI THỰC HÀNH SỐ 2

## MARKOV DECISION PROCESS, HÀM GIÁ TRỊ VÀ QUY HOẠCH ĐỘNG

**Học phần:** Học tăng cường – Reinforcement Learning  
**Ngôn ngữ lập trình:** Python  
**Số lượng bài tập:** 36 bài  
**Hình thức:** Thực hành lập trình cá nhân  
**Thời lượng đề nghị:** 6–8 tiết trên lớp + phần hoàn thiện ở nhà  
**Hình thức nộp bài:** Cập nhật vào repository GitHub cá nhân đã dùng ở Lab01  
**Folder nộp bài:** `Lab02/`

---

# 1. Chủ đề của bài thực hành

Lab02 chuyển từ việc **tương tác với môi trường** ở Lab01 sang việc **mô hình hóa và giải bài toán học tăng cường bằng Markov Decision Process (MDP)**.

Các nội dung chính:

1. Markov property.
2. Markov chain.
3. Markov Decision Process.
4. Transition probability.
5. Reward và Return.
6. Discount factor.
7. Policy.
8. State-value function `V(s)`.
9. State-action value function `Q(s,a)`.
10. Bellman expectation equation.
11. Bellman optimality equation.
12. Dynamic Programming.
13. Policy Evaluation.
14. Policy Improvement.
15. Policy Iteration.
16. Value Iteration.
17. Giải `FrozenLake-v1` bằng Dynamic Programming.

> **Yêu cầu trọng tâm:** Sinh viên phải tự lập trình các thuật toán Dynamic Programming. Không sử dụng thư viện RL bên ngoài để gọi sẵn `value_iteration()`, `policy_iteration()` hoặc thuật toán tương đương.

---

# 2. Mục tiêu

Sau khi hoàn thành Lab02, sinh viên phải có khả năng:

1. Biểu diễn Markov chain bằng ma trận xác suất chuyển trạng thái.
2. Kiểm tra tính hợp lệ của transition matrix bằng code.
3. Mô phỏng Markov chain bằng NumPy.
4. Biểu diễn một MDP rời rạc bằng Python.
5. Phân biệt state, action, transition probability và reward.
6. Tính return từ một chuỗi reward.
7. Phân tích ảnh hưởng của discount factor `gamma`.
8. Biểu diễn deterministic policy và stochastic policy.
9. Tính state-value function.
10. Tính state-action value function.
11. Cài đặt Bellman backup.
12. Đọc model chuyển trạng thái của `FrozenLake-v1`.
13. Cài đặt Iterative Policy Evaluation.
14. Cài đặt Policy Improvement.
15. Cài đặt Policy Iteration.
16. Cài đặt Value Iteration.
17. Trích xuất optimal policy.
18. Đánh giá policy bằng nhiều episode.
19. So sánh Value Iteration và Policy Iteration.
20. Tổ chức một chương trình Dynamic Programming hoàn chỉnh.

---

# 3. Kiến thức chuẩn bị

Sinh viên cần hoàn thành Lab01 và ôn lại:

- xác suất cơ bản;
- vector, ma trận;
- NumPy;
- function, list, dictionary;
- vòng lặp Python;
- Gymnasium;
- `reset()` và `step()`;
- state, action, reward, episode, policy.

---

# 4. Công cụ và thư viện

Tiếp tục sử dụng stack của Lab01:

- Python 3.12 trở lên;
- Gymnasium 1.3.0;
- NumPy;
- Matplotlib;
- Jupyter Notebook hoặc VS Code;
- Git;
- GitHub.

Cài đặt:

```bash
pip install "gymnasium[toy-text]==1.3.0"
pip install numpy matplotlib jupyter
```

Kiểm tra:

```bash
python --version
pip show gymnasium
pip show numpy
```

---

# 5. Cấu trúc repository

```text
RL_MSSV_HoTen/
│
├── README.md
├── Lab01/
│   └── ...
│
├── Lab02/
│   ├── README.md
│   ├── requirements.txt
│   ├── src/
│   ├── notebooks/
│   ├── figures/
│   └── data/
│
└── ...
```

Cấu trúc bắt buộc:

```text
Lab02/
│
├── README.md
├── requirements.txt
├── src/
│   ├── bai01.py
│   ├── bai02.py
│   ├── ...
│   ├── bai36.py
│   ├── mdp_utils.py
│   └── main.py
│
├── notebooks/
│   └── Lab02_MSSV_HoTen.ipynb
│
├── figures/
│   ├── markov_distribution.png
│   ├── gamma_comparison.png
│   ├── value_iteration_convergence.png
│   ├── policy_iteration_convergence.png
│   └── algorithm_comparison.png
│
└── data/
    └── README.md
```

---

# 6. Công thức cần sử dụng

## 6.1. Return

Với chuỗi reward `R_(t+1), R_(t+2), ...`, discounted return:

```text
G_t = R_(t+1) + gamma*R_(t+2) + gamma^2*R_(t+3) + ...
```

Trong đó:

```text
0 <= gamma <= 1
```

## 6.2. State-value function

```text
V_pi(s) = E_pi[G_t | S_t = s]
```

## 6.3. State-action value function

```text
Q_pi(s,a) = E_pi[G_t | S_t = s, A_t = a]
```

## 6.4. Bellman expectation backup

```text
V_pi(s) = sum_a pi(a|s) * sum_(s',r) p(s',r|s,a) * [r + gamma*V_pi(s')]
```

## 6.5. Bellman optimality backup

```text
V*(s) = max_a sum_(s',r) p(s',r|s,a) * [r + gamma*V*(s')]
```

---

# 7. Làm quen model của FrozenLake

```python
import gymnasium as gym

env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True,
)
```

Số state:

```python
n_states = env.observation_space.n
```

Số action:

```python
n_actions = env.action_space.n
```

Để làm Dynamic Programming, truy cập model:

```python
P = env.unwrapped.P
```

Duyệt transition:

```python
for probability, next_state, reward, terminated in P[state][action]:
    print(probability, next_state, reward, terminated)
```

> Không được giả định mỗi `(state, action)` chỉ có một `next_state`. Khi môi trường stochastic, một action có thể dẫn đến nhiều state kế tiếp với các xác suất khác nhau.

---

# 8. Quy ước action FrozenLake

Sinh viên kiểm tra và xây dựng:

```python
ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP",
}

ACTION_SYMBOLS = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑",
}
```

---

# 9. BÀI TẬP

# PHẦN A — Markov Chain

## Bài 1. Tạo transition matrix

Cho ba trạng thái:

```text
Sunny
Cloudy
Rainy
```

Tự xây dựng transition matrix `P` kích thước `3 x 3`:

```python
P = np.array([
    [...],
    [...],
    [...],
])
```

Mỗi hàng phải có tổng bằng 1. In ma trận ra màn hình.

Lưu tại:

```text
Lab02/src/bai01.py
```

## Bài 2. Kiểm tra transition matrix

Viết:

```python
def validate_transition_matrix(P, tol=1e-10):
    ...
```

Kiểm tra:

1. `P` là ma trận vuông;
2. mọi phần tử thuộc `[0,1]`;
3. tổng mỗi hàng xấp xỉ 1.

Trả về `True` hoặc `False`.

## Bài 3. Tính xác suất trạng thái kế tiếp

Cho:

```python
p0 = np.array([1.0, 0.0, 0.0])
```

Tính distribution sau một bước. Không hard-code kết quả.

## Bài 4. Distribution sau nhiều bước

Viết:

```python
def state_distribution(p0, P, n_steps):
    ...
```

Tính distribution tại:

```text
t = 1, 2, 5, 10, 50
```

## Bài 5. Mô phỏng Markov chain

Viết:

```python
def sample_next_state(current_state, P, rng):
    ...
```

Dùng `rng.choice(...)`, mô phỏng 30 transition và in chuỗi state.

## Bài 6. So sánh lý thuyết và mô phỏng

Mô phỏng ít nhất `100000` transition. Tính tần suất state và so sánh với distribution theo phép nhân ma trận sau nhiều bước.

Viết nhận xét 3–5 dòng.

---

# PHẦN B — Reward, Return và Discount Factor

## Bài 7. Undiscounted return

Cho:

```python
rewards = [1, 1, 1, 1, 1]
```

Viết:

```python
def compute_return(rewards, gamma):
    ...
```

Tính với `gamma = 1.0`.

## Bài 8. Discounted return

Tính với:

```text
gamma = 0.0, 0.5, 0.9, 0.99, 1.0
```

Lập bảng:

| Gamma | Return |
|---:|---:|
| 0.00 | |
| 0.50 | |
| 0.90 | |
| 0.99 | |
| 1.00 | |

## Bài 9. Return từ cuối episode

Viết:

```python
def discounted_returns(rewards, gamma):
    ...
```

Với:

```python
rewards = [0, 0, 0, 1]
```

Output chứa `G_0, G_1, G_2, G_3`. Tính theo chiều từ cuối episode về đầu.

## Bài 10. Ảnh hưởng của gamma

Cho:

```python
rewards = [0, 0, 0, 0, 10]
gammas = np.linspace(0, 1, 101)
```

Vẽ `G_0` theo `gamma` và lưu:

```text
Lab02/figures/gamma_comparison.png
```

## Bài 11. Reward sớm và reward trễ

So sánh:

```python
sequence_A = [5, 0, 0, 0, 0]
sequence_B = [0, 0, 0, 0, 10]
```

Dùng code tìm khoảng `gamma` mà B có return lớn hơn A.

---

# PHẦN C — Mô hình hóa một MDP nhỏ

## Bài 12. Xây dựng MDP hai state

Tạo MDP gồm `State 0`, `State 1`, `Action 0`, `Action 1`.

Biểu diễn:

```python
P[state][action] = [
    (probability, next_state, reward, terminated)
]
```

Tự thiết kế xác suất và reward hợp lệ.

## Bài 13. Kiểm tra model MDP

Viết:

```python
def validate_mdp(P, n_states, n_actions):
    ...
```

Kiểm tra tổng xác suất transition của từng `(state, action)` bằng 1.

Nếu sai:

```text
Invalid transition at state=..., action=...
```

## Bài 14. Deterministic policy

Tạo:

```python
policy = np.array([...])
```

Viết:

```python
def print_policy(policy):
    ...
```

## Bài 15. Stochastic policy

Tạo uniform random policy:

```python
policy = np.ones((n_states, n_actions)) / n_actions
```

Viết code kiểm tra tổng xác suất action tại mỗi state bằng 1.

---

# PHẦN D — Khám phá model FrozenLake

## Bài 16. Thông tin cơ bản

Tạo `FrozenLake-v1`, `map_name="4x4"`, `is_slippery=True`.

In:

```text
Number of states
Number of actions
Initial observation
```

## Bài 17. In transition model của một state

Với `state = 0`, in toàn bộ transition theo từng action:

```text
Action
Probability
Next state
Reward
Terminated
```

## Bài 18. Hàm `describe_state()`

Viết:

```python
def describe_state(env, state):
    ...
```

Kiểm thử với state `0`, `1`, `14`.

## Bài 19. Kiểm tra tổng xác suất transition

Duyệt mọi state/action và xác nhận:

```python
np.isclose(sum(probabilities), 1.0)
```

## Bài 20. Deterministic và stochastic FrozenLake

So sánh cùng:

```text
state = 0
action = RIGHT
```

trong hai environment:

```python
is_slippery=False
is_slippery=True
```

In số transition và xác suất. Viết kết luận 3–5 dòng.

---

# PHẦN E — Bellman Backup và Policy Evaluation

## Bài 21. Một Bellman backup

Viết:

```python
def q_from_v(env, V, state, action, gamma):
    ...
```

Tính:

```text
Q(s,a) = sum p * [reward + gamma * V(next_state)]
```

Phải duyệt mọi transition.

## Bài 22. Tính toàn bộ Q của một state

Viết:

```python
def action_values(env, V, state, gamma):
    ...
```

Trả về vector kích thước `env.action_space.n`.

Kiểm thử với:

```python
V = np.zeros(env.observation_space.n)
```

## Bài 23. Một sweep của Policy Evaluation

Cho uniform random policy.

Viết:

```python
def policy_evaluation_sweep(env, policy, V, gamma):
    ...
```

Chỉ thực hiện một sweep qua tất cả state.

## Bài 24. Iterative Policy Evaluation

Viết:

```python
def policy_evaluation(
    env,
    policy,
    gamma=0.99,
    theta=1e-8,
    max_iterations=10000,
):
    ...
```

Dừng khi:

```python
delta < theta
```

Trả về:

```python
V, n_iterations
```

## Bài 25. Theo dõi hội tụ

Mở rộng Bài 24, lưu:

```python
deltas = []
```

Mỗi iteration:

```python
delta = np.max(np.abs(new_V - V))
```

Vẽ `delta` theo iteration.

---

# PHẦN F — Policy Improvement và Policy Iteration

## Bài 26. Greedy policy từ V

Viết:

```python
def greedy_policy_from_value(env, V, gamma=0.99):
    ...
```

Với mỗi state:

1. tính `Q(s,a)` cho mọi action;
2. chọn `np.argmax(q_values)`;
3. lưu action vào policy.

## Bài 27. Hiển thị policy trên lưới 4x4

Viết:

```python
def print_frozenlake_policy(env, policy):
    ...
```

Dùng mũi tên:

```text
← ↓ → ↑
```

State Hole hiển thị `H`, Goal hiển thị `G`.

## Bài 28. Một bước Policy Improvement

Cho một policy ban đầu:

1. Policy Evaluation;
2. Greedy Policy Improvement;
3. so sánh `old_policy` và `new_policy`;
4. đếm số state đổi action.

## Bài 29. Cài đặt Policy Iteration

Viết:

```python
def policy_iteration(
    env,
    gamma=0.99,
    theta=1e-8,
    max_iterations=1000,
):
    ...
```

Quy trình:

```text
Khởi tạo policy
       ↓
Policy Evaluation
       ↓
Policy Improvement
       ↓
Policy stable?
   ↙         ↘
 No         Yes
 ↓           ↓
lặp       kết thúc
```

Trả về:

```python
policy, V, n_policy_iterations
```

## Bài 30. Kiểm tra policy stability

Tự lập trình:

```python
policy_stable = ...
```

In:

```text
Policy Iteration converged after ... iterations.
```

---

# PHẦN G — Value Iteration

## Bài 31. Một sweep của Value Iteration

Viết:

```python
def value_iteration_sweep(env, V, gamma):
    ...
```

Mỗi state:

```python
new_V[state] = np.max(q_values)
```

## Bài 32. Value Iteration hoàn chỉnh

Viết:

```python
def value_iteration(
    env,
    gamma=0.99,
    theta=1e-8,
    max_iterations=10000,
):
    ...
```

Trả về:

```python
V, n_iterations, deltas
```

Dừng khi:

```python
delta < theta
```

## Bài 33. Trích xuất optimal policy

Sau Value Iteration:

```python
optimal_policy = greedy_policy_from_value(env, V, gamma)
```

In:

```text
Optimal state values
Optimal policy
```

Hiển thị policy dạng 4x4.

---

# PHẦN H — Đánh giá và so sánh thuật toán

## Bài 34. Đánh giá policy bằng simulation

Viết:

```python
def evaluate_policy_by_simulation(
    env,
    policy,
    n_episodes=1000,
    seed=42,
):
    ...
```

Action được lấy bằng:

```python
action = policy[state]
```

Thu thập:

- success rate;
- mean reward;
- mean episode length;
- min episode length;
- max episode length.

Đánh giá:

1. random policy;
2. policy từ Value Iteration;
3. policy từ Policy Iteration.

## Bài 35. So sánh Value Iteration và Policy Iteration

Chạy cả hai trên:

```python
FrozenLake-v1
map_name="4x4"
is_slippery=True
```

Đo thời gian bằng:

```python
from time import perf_counter
```

Lập bảng:

| Thuật toán | Số vòng lặp | Thời gian | Success rate | Mean reward |
|---|---:|---:|---:|---:|
| Value Iteration | | | | |
| Policy Iteration | | | | |

Lưu biểu đồ:

```text
Lab02/figures/algorithm_comparison.png
```

Viết nhận xét tối thiểu 8 dòng.

## Bài 36. Mini-project: Dynamic Programming Solver

Xây dựng chương trình hoàn chỉnh:

```python
create_environment()
get_transition_model()
q_from_v()
policy_evaluation()
greedy_policy_from_value()
policy_iteration()
value_iteration()
evaluate_policy_by_simulation()
print_policy()
plot_convergence()
main()
```

Yêu cầu:

1. chạy được `FrozenLake-v1`;
2. hỗ trợ `is_slippery=False`;
3. hỗ trợ `is_slippery=True`;
4. có tham số `gamma`;
5. có tham số `theta`;
6. có `max_iterations`;
7. cài đặt Value Iteration;
8. cài đặt Policy Iteration;
9. hiển thị value table;
10. hiển thị policy dạng lưới;
11. đánh giá bằng ít nhất 1000 episode;
12. in success rate;
13. in mean reward;
14. đo runtime;
15. lưu dữ liệu hội tụ;
16. vẽ convergence curve;
17. so sánh hai thuật toán;
18. có comment/docstring;
19. không dùng thuật toán DP có sẵn từ thư viện RL;
20. cập nhật `README.md`.

Lưu:

```text
Lab02/src/main.py
```

---

# 10. Bài mở rộng tự chọn

## Mở rộng 1. FrozenLake 8x8

Thay:

```python
map_name="4x4"
```

bằng:

```python
map_name="8x8"
```

So sánh:

- số state;
- số iteration;
- runtime;
- success rate.

## Mở rộng 2. Ảnh hưởng của gamma

Thử:

```text
gamma = 0.50
0.80
0.90
0.95
0.99
1.00
```

So sánh value function, policy và success rate.

## Mở rộng 3. Ảnh hưởng của theta

Thử:

```text
theta = 1e-2
1e-4
1e-6
1e-8
1e-10
```

Đo số iteration và runtime.

---

# 11. Skeleton code khuyến nghị

```python
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


ACTION_SYMBOLS = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑",
}


def q_from_v(env, V, state, action, gamma):
    # TODO
    pass


def policy_evaluation(
    env,
    policy,
    gamma=0.99,
    theta=1e-8,
    max_iterations=10000,
):
    # TODO
    pass


def greedy_policy_from_value(env, V, gamma=0.99):
    # TODO
    pass


def policy_iteration(
    env,
    gamma=0.99,
    theta=1e-8,
    max_iterations=1000,
):
    # TODO
    pass


def value_iteration(
    env,
    gamma=0.99,
    theta=1e-8,
    max_iterations=10000,
):
    # TODO
    pass


def evaluate_policy_by_simulation(
    env,
    policy,
    n_episodes=1000,
    seed=42,
):
    # TODO
    pass


def print_policy(env, policy):
    # TODO
    pass


def main():
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=True,
    )

    # TODO: Value Iteration
    # TODO: Policy Iteration
    # TODO: Evaluate policies
    # TODO: Compare algorithms
    # TODO: Plot convergence

    env.close()


if __name__ == "__main__":
    main()
```

---

# 12. Yêu cầu lập trình

## 12.1. Không dùng thuật toán có sẵn

Không được gọi thư viện thực hiện sẵn toàn bộ Value Iteration hoặc Policy Iteration.

Mục tiêu của Lab02 là tự lập trình:

```text
Bellman backup
Policy Evaluation
Policy Improvement
Policy Iteration
Value Iteration
```

## 12.2. Không hard-code optimal policy

Không được viết một mảng optimal policy cố định rồi xem là kết quả thuật toán.

Policy phải được sinh ra từ value function hoặc Policy Iteration.

## 12.3. Không bỏ qua transition probability

Không được chỉ tính:

```python
value = reward + gamma * V[next_state]
```

nếu action có nhiều transition.

Phải duyệt và cộng theo xác suất:

```python
value += probability * (
    reward + gamma * V[next_state]
)
```

## 12.4. Terminal transition

Sinh viên phải kiểm tra biến `terminated` trong transition model và giải thích cách xử lý terminal transition trong notebook.

---

# 13. Câu hỏi phải trả lời trong notebook

1. Markov property là gì?
2. Markov chain khác MDP như thế nào?
3. Transition probability là gì?
4. Vì sao tổng transition probability của một `(state, action)` phải bằng 1?
5. Return khác immediate reward như thế nào?
6. Discount factor có vai trò gì?
7. Khi `gamma = 0`, agent quan tâm điều gì?
8. Khi `gamma` gần 1, reward xa trong tương lai ảnh hưởng thế nào?
9. Policy là gì?
10. Deterministic policy khác stochastic policy thế nào?
11. `V(s)` biểu diễn điều gì?
12. `Q(s,a)` biểu diễn điều gì?
13. Bellman equation có tính đệ quy ở điểm nào?
14. Bellman expectation khác Bellman optimality thế nào?
15. Dynamic Programming cần biết thông tin gì về môi trường?
16. Policy Evaluation dùng để làm gì?
17. Policy Improvement dùng để làm gì?
18. Policy Iteration hoạt động như thế nào?
19. Value Iteration hoạt động như thế nào?
20. Value Iteration khác Policy Iteration ở điểm nào?
21. Vì sao `FrozenLake-v1` thích hợp để minh họa Dynamic Programming?
22. `is_slippery=True` làm model thay đổi như thế nào?
23. `theta` ảnh hưởng thế nào đến số iteration?
24. Vì sao cần đánh giá optimal policy bằng simulation?
25. Nếu không biết transition model, Dynamic Programming có áp dụng trực tiếp được không? Giải thích.

---

# 14. File `mdp_utils.py`

Tạo:

```text
Lab02/src/mdp_utils.py
```

Tối thiểu chứa:

```python
def q_from_v(...):
    ...


def policy_evaluation(...):
    ...


def greedy_policy_from_value(...):
    ...


def policy_iteration(...):
    ...


def value_iteration(...):
    ...


def evaluate_policy_by_simulation(...):
    ...
```

Các bài cuối phải import lại thay vì copy-paste code.

---

# 15. Yêu cầu biểu đồ

Tối thiểu:

```text
Lab02/figures/
├── markov_distribution.png
├── gamma_comparison.png
├── value_iteration_convergence.png
├── policy_iteration_convergence.png
└── algorithm_comparison.png
```

Tất cả biểu đồ phải có title, xlabel, ylabel, legend khi cần và grid khi phù hợp.

---

# 16. `requirements.txt`

```text
gymnasium[toy-text]==1.3.0
numpy
matplotlib
jupyter
```

---

# 17. `Lab02/README.md`

README phải có:

```markdown
# Lab02 - Markov Decision Process và Dynamic Programming

## Thông tin sinh viên

- Họ tên:
- MSSV:
- Lớp:
- GitHub username:

## Mục tiêu

## Cấu trúc thư mục

## Cài đặt

## Cách chạy

## Thuật toán đã cài đặt

### Policy Evaluation

### Policy Iteration

### Value Iteration

## Kết quả FrozenLake

## So sánh Value Iteration và Policy Iteration

## Nhận xét

## Tài liệu tham khảo
```

---

# 18. Cách chạy

```bash
cd Lab02
pip install -r requirements.txt
python src/bai01.py
python src/bai24.py
python src/bai29.py
python src/bai32.py
python src/main.py
```

Notebook:

```bash
jupyter notebook
```

---

# 19. Quy định GitHub

Khuyến nghị lịch sử commit:

```text
Khoi tao Lab02
Hoan thanh Markov chain exercises
Them discounted return experiments
Explore FrozenLake transition model
Implement policy evaluation
Implement policy improvement
Implement policy iteration
Implement value iteration
Add convergence plots
Compare DP algorithms
Complete Lab02 report
```

Ví dụ:

```bash
git add .
git commit -m "Implement value iteration"
git push origin main
```

---

# 20. Thang điểm đề nghị

| Nội dung | Điểm |
|---|---:|
| Markov chain – Bài 1–6 | 1.0 |
| Reward, Return, Gamma – Bài 7–11 | 1.0 |
| MDP representation – Bài 12–15 | 0.75 |
| FrozenLake model – Bài 16–20 | 1.0 |
| Bellman + Policy Evaluation – Bài 21–25 | 1.5 |
| Policy Improvement + Policy Iteration – Bài 26–30 | 1.5 |
| Value Iteration – Bài 31–33 | 1.25 |
| Đánh giá + so sánh – Bài 34–35 | 1.0 |
| Mini-project – Bài 36 | 0.75 |
| Tổ chức code, README, GitHub | 0.25 |
| **Tổng** | **10.0** |

Điều kiện:

- Code không chạy: phần tương ứng tối đa 50% số điểm.
- Hard-code optimal policy: không tính phần thuật toán tương ứng.
- Dùng thư viện gọi sẵn Value Iteration/Policy Iteration: không tính phần thuật toán tương ứng.
- Bỏ qua transition probability: yêu cầu sửa.
- Không có simulation để đánh giá: thiếu phần đánh giá.
- Không có biểu đồ yêu cầu: trừ điểm.
- Không cập nhật `Lab02/README.md`: trừ điểm.
- Folder không đúng cấu trúc: trừ điểm tổ chức.
- Link GitHub không truy cập được: có thể xem là chưa nộp.

---

# 21. Phân mức bài tập

**Mức 1 — Markov và Return:** Bài 1 → Bài 11  
Mục tiêu: Markov chain, transition probability, return, gamma.

**Mức 2 — MDP Model:** Bài 12 → Bài 20  
Mục tiêu: biểu diễn MDP và đọc transition model.

**Mức 3 — Bellman và Evaluation:** Bài 21 → Bài 25  
Mục tiêu: Bellman backup và value function của một policy.

**Mức 4 — Dynamic Programming:** Bài 26 → Bài 33  
Mục tiêu: Policy Improvement, Policy Iteration, Value Iteration, optimal policy.

**Mức 5 — Tổng hợp:** Bài 34 → Bài 36  
Mục tiêu: đánh giá, so sánh và xây dựng pipeline DP hoàn chỉnh.

---

# 22. Kết quả mong đợi

Sau Lab02, sinh viên phải hiểu chuỗi logic:

```text
Environment model
       ↓
Transition probabilities + Rewards
       ↓
Bellman equation
       ↓
Value function
       ↓
Policy improvement
       ↓
Optimal policy
```

Sinh viên phải tự lập trình được:

```python
def q_from_v(...):
    ...


def policy_evaluation(...):
    ...


def greedy_policy_from_value(...):
    ...


def policy_iteration(...):
    ...


def value_iteration(...):
    ...
```

Và phải hiểu vì sao Dynamic Programming có thể giải `FrozenLake-v1` khi transition dynamics của môi trường được biết.

---

# 23. Liên hệ với bài thực hành tiếp theo

Lab02 sử dụng **model-based Dynamic Programming**, tức thuật toán cần biết transition dynamics của môi trường.

Các bài tiếp theo sẽ chuyển sang trường hợp agent học từ episode/experience mà không cần biết trước đầy đủ model môi trường.

Sinh viên cần nắm chắc:

```text
Return
Value
Q-value
Bellman equation
Policy
```

trước khi chuyển sang Monte Carlo, Temporal-Difference Learning, SARSA và Q-Learning.
