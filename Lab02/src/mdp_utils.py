"""Thu vien dung chung cho Lab02 - MDP va Dynamic Programming.

Moi thuat toan Dynamic Programming trong Lab02 duoc cai dat o day (mot lan duy
nhat) theo dung yeu cau muc 14 cua de bai; cac file baiXX.py import lai thay vi
copy-paste code.

Noi dung:
    create_environment()            - tao FrozenLake va lay model
    get_transition_model()          - lay P[state][action]
    q_from_v()                      - mot Bellman backup cho (state, action)
    action_values()                 - vector Q(s, .) cua mot state
    policy_evaluation_sweep()       - mot sweep cua Policy Evaluation
    policy_evaluation()             - Iterative Policy Evaluation
    greedy_policy_from_value()      - Policy Improvement (greedy theo V)
    policy_iteration()              - Policy Iteration
    value_iteration_sweep()         - mot sweep cua Value Iteration
    value_iteration()               - Value Iteration
    evaluate_policy_by_simulation() - danh gia policy bang mo phong
    print_policy_grid()             - in policy dang luoi voi mui ten

Khong dung bat ky thu vien RL nao co san thuat toan DP: tat ca deu tu viet
bang NumPy va vong lap Python.
"""

import sys

import gymnasium as gym
import numpy as np

# Windows: khi stdout bi redirect ra file/pipe thi mac dinh la cp1252 va khong
# in duoc mui ten Unicode. Chuyen sang UTF-8 neu co the.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP",
}

ACTION_SYMBOLS = {
    0: "←",   # <-
    1: "↓",   # v
    2: "→",   # ->
    3: "↑",   # ^
}

# Ban du phong khi terminal khong hien thi duoc ky tu Unicode
ACTION_SYMBOLS_ASCII = {
    0: "<",
    1: "v",
    2: ">",
    3: "^",
}


def action_symbols() -> dict[int, str]:
    """Tra ve bang ky hieu action dung duoc voi terminal hien tai."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "".join(ACTION_SYMBOLS.values()).encode(encoding)
        return ACTION_SYMBOLS
    except (UnicodeEncodeError, LookupError):
        return ACTION_SYMBOLS_ASCII


def create_environment(map_name: str = "4x4",
                       is_slippery: bool = True,
                       render_mode=None) -> gym.Env:
    """Tao moi truong FrozenLake-v1 dung cho Dynamic Programming."""
    return gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode=render_mode,
    )


def get_transition_model(env: gym.Env) -> dict:
    """Lay model chuyen trang thai P[state][action] cua moi truong.

    Moi phan tu la list cac tuple (probability, next_state, reward, terminated).
    """
    return env.unwrapped.P


def env_sizes(env: gym.Env) -> tuple[int, int]:
    """Tra ve (n_states, n_actions) doc truc tiep tu cac space."""
    return int(env.observation_space.n), int(env.action_space.n)


def q_from_v(env: gym.Env, V, state: int, action: int, gamma: float = 0.99) -> float:
    """Mot Bellman backup: Q(s,a) = sum_s' p(s'|s,a) * [r + gamma * V(s')].

    Duyet TAT CA transition cua (state, action) va cong theo xac suat - khong
    duoc gia dinh moi cap (state, action) chi co mot next_state.

    Neu transition dan toi trang thai ket thuc (terminated=True) thi khong
    bootstrap tiep: gia tri tuong lai cua trang thai do bang 0.
    """
    transition_model = get_transition_model(env)

    q_value = 0.0
    for probability, next_state, reward, terminated in transition_model[state][action]:
        future_value = 0.0 if terminated else gamma * V[next_state]
        q_value += probability * (reward + future_value)

    return float(q_value)


def action_values(env: gym.Env, V, state: int, gamma: float = 0.99) -> np.ndarray:
    """Tinh vector Q(state, .) cho moi action cua mot state."""
    _, n_actions = env_sizes(env)
    return np.array([q_from_v(env, V, state, action, gamma)
                     for action in range(n_actions)], dtype=np.float64)


def as_stochastic_policy(policy, n_states: int, n_actions: int) -> np.ndarray:
    """Chuan hoa policy ve dang stochastic (n_states, n_actions).

    Chap nhan ca policy deterministic (mang 1 chieu chua chi so action) lan
    policy stochastic (ma tran xac suat).
    """
    policy_array = np.asarray(policy)

    if policy_array.ndim == 1:
        # Deterministic -> one-hot
        stochastic = np.zeros((n_states, n_actions), dtype=np.float64)
        stochastic[np.arange(n_states), policy_array.astype(int)] = 1.0
        return stochastic

    return policy_array.astype(np.float64)


def uniform_random_policy(n_states: int, n_actions: int) -> np.ndarray:
    """Stochastic policy chon deu moi action."""
    return np.ones((n_states, n_actions), dtype=np.float64) / n_actions


def policy_evaluation_sweep(env: gym.Env, policy, V, gamma: float = 0.99) -> np.ndarray:
    """Mot sweep cua Policy Evaluation qua toan bo state.

    V_pi(s) <- sum_a pi(a|s) * sum_s' p(s'|s,a) * [r + gamma * V_pi(s')]
    """
    n_states, n_actions = env_sizes(env)
    stochastic_policy = as_stochastic_policy(policy, n_states, n_actions)

    new_V = np.zeros(n_states, dtype=np.float64)
    for state in range(n_states):
        q_values = action_values(env, V, state, gamma)
        new_V[state] = float(np.dot(stochastic_policy[state], q_values))

    return new_V


def policy_evaluation(env: gym.Env,
                      policy,
                      gamma: float = 0.99,
                      theta: float = 1e-8,
                      max_iterations: int = 10000,
                      track_deltas: bool = False):
    """Iterative Policy Evaluation.

    Lap lai Bellman expectation backup cho den khi delta < theta.

    Tra ve (V, n_iterations), hoac (V, n_iterations, deltas) neu
    track_deltas=True.
    """
    n_states, _ = env_sizes(env)

    V = np.zeros(n_states, dtype=np.float64)
    deltas = []
    n_iterations = 0

    for iteration in range(1, max_iterations + 1):
        new_V = policy_evaluation_sweep(env, policy, V, gamma)

        delta = float(np.max(np.abs(new_V - V)))
        deltas.append(delta)

        V = new_V
        n_iterations = iteration

        if delta < theta:
            break

    if track_deltas:
        return V, n_iterations, deltas
    return V, n_iterations


def greedy_policy_from_value(env: gym.Env, V, gamma: float = 0.99) -> np.ndarray:
    """Policy Improvement: chon action tham lam theo V.

    Voi moi state: tinh Q(s,a) cho moi action roi lay np.argmax.
    Tra ve policy deterministic dang mang chi so action.
    """
    n_states, _ = env_sizes(env)

    policy = np.zeros(n_states, dtype=np.int64)
    for state in range(n_states):
        q_values = action_values(env, V, state, gamma)
        policy[state] = int(np.argmax(q_values))

    return policy


def policy_iteration(env: gym.Env,
                     gamma: float = 0.99,
                     theta: float = 1e-8,
                     max_iterations: int = 1000,
                     track_history: bool = False):
    """Policy Iteration: lap Policy Evaluation + Policy Improvement.

    Khoi tao policy -> Policy Evaluation -> Policy Improvement -> policy on
    dinh? -> neu chua thi lap lai.

    Tra ve (policy, V, n_policy_iterations); neu track_history=True thi tra
    them mot dictionary lich su de ve bieu do hoi tu.
    """
    n_states, n_actions = env_sizes(env)

    # Khoi tao bang policy deterministic tuy y (luon chon action 0)
    policy = np.zeros(n_states, dtype=np.int64)
    V = np.zeros(n_states, dtype=np.float64)

    history = {"mean_value": [], "n_changed": [], "eval_iterations": []}
    n_policy_iterations = 0

    for iteration in range(1, max_iterations + 1):
        # 1) Policy Evaluation cho policy hien tai
        V, eval_iterations = policy_evaluation(env, policy, gamma, theta)

        # 2) Policy Improvement
        new_policy = greedy_policy_from_value(env, V, gamma)

        # 3) Kiem tra tinh on dinh cua policy (tu lap trinh, khong dung ham san)
        n_changed = int(np.sum(new_policy != policy))
        policy_stable = n_changed == 0

        history["mean_value"].append(float(np.mean(V)))
        history["n_changed"].append(n_changed)
        history["eval_iterations"].append(eval_iterations)

        policy = new_policy
        n_policy_iterations = iteration

        if policy_stable:
            break

    if track_history:
        return policy, V, n_policy_iterations, history
    return policy, V, n_policy_iterations


def value_iteration_sweep(env: gym.Env, V, gamma: float = 0.99) -> np.ndarray:
    """Mot sweep cua Value Iteration: new_V[s] = max_a Q(s,a)."""
    n_states, _ = env_sizes(env)

    new_V = np.zeros(n_states, dtype=np.float64)
    for state in range(n_states):
        q_values = action_values(env, V, state, gamma)
        new_V[state] = float(np.max(q_values))

    return new_V


def value_iteration(env: gym.Env,
                    gamma: float = 0.99,
                    theta: float = 1e-8,
                    max_iterations: int = 10000):
    """Value Iteration: lap Bellman optimality backup den khi delta < theta.

    Tra ve (V, n_iterations, deltas).
    """
    n_states, _ = env_sizes(env)

    V = np.zeros(n_states, dtype=np.float64)
    deltas = []
    n_iterations = 0

    for iteration in range(1, max_iterations + 1):
        new_V = value_iteration_sweep(env, V, gamma)

        delta = float(np.max(np.abs(new_V - V)))
        deltas.append(delta)

        V = new_V
        n_iterations = iteration

        if delta < theta:
            break

    return V, n_iterations, deltas


def evaluate_policy_by_simulation(env: gym.Env,
                                  policy,
                                  n_episodes: int = 1000,
                                  seed: int = 42,
                                  max_steps: int = 1000) -> dict:
    """Danh gia mot policy deterministic bang mo phong that tren moi truong.

    Action duoc lay bang: action = policy[state].
    Tra ve success rate, mean reward, mean/min/max episode length.
    """
    policy_array = np.asarray(policy)
    if policy_array.ndim == 2:
        policy_array = np.argmax(policy_array, axis=1)

    # Seed mot lan o dau thi nghiem de toan bo mo phong tai lap duoc
    env.reset(seed=seed)
    env.action_space.seed(seed)

    rewards = []
    lengths = []

    for _ in range(n_episodes):
        state, info = env.reset()
        total_reward = 0.0
        length = 0

        while length < max_steps:
            action = int(policy_array[int(state)])
            state, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            length += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(length)

    rewards_array = np.array(rewards, dtype=np.float64)
    lengths_array = np.array(lengths, dtype=np.float64)

    return {
        "n_episodes": n_episodes,
        "success": int((rewards_array > 0).sum()),
        "success_rate": float((rewards_array > 0).mean()),
        "mean_reward": float(rewards_array.mean()),
        "mean_length": float(lengths_array.mean()),
        "min_length": int(lengths_array.min()),
        "max_length": int(lengths_array.max()),
    }


def print_value_grid(env: gym.Env, V, decimals: int = 4) -> None:
    """In state-value function duoi dang luoi giong ban do FrozenLake."""
    desc = env.unwrapped.desc
    n_rows, n_cols = desc.shape

    for row in range(n_rows):
        cells = []
        for col in range(n_cols):
            cells.append(f"{V[row * n_cols + col]:{decimals + 3}.{decimals}f}")
        print("  " + " ".join(cells))


def print_policy_grid(env: gym.Env, policy) -> None:
    """In policy duoi dang luoi: mui ten cho o di duoc, H cho Hole, G cho Goal."""
    policy_array = np.asarray(policy)
    if policy_array.ndim == 2:
        policy_array = np.argmax(policy_array, axis=1)

    symbols = action_symbols()
    desc = env.unwrapped.desc
    n_rows, n_cols = desc.shape

    for row in range(n_rows):
        cells = []
        for col in range(n_cols):
            state = row * n_cols + col
            cell_type = desc[row][col].decode("ascii")

            if cell_type == "H":
                cells.append("H")       # Hole: khong can action
            elif cell_type == "G":
                cells.append("G")       # Goal: trang thai ket thuc
            else:
                cells.append(symbols[int(policy_array[state])])

        print("  " + " ".join(cells))


def print_policy(env: gym.Env, policy) -> None:
    """Ten goi theo skeleton cua de bai (muc 11), goi lai print_policy_grid()."""
    print_policy_grid(env, policy)
