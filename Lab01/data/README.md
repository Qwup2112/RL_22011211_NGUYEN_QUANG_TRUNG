# Thư mục `data/`

Nơi lưu dữ liệu thí nghiệm sinh ra khi chạy code trong `Lab01/src/`.

| File | Sinh ra bởi | Nội dung |
|------|-------------|----------|
| `random_agent_100_episodes.csv` | `python src/bai14.py` | reward và length của 100 episode chạy bằng random agent trên `CartPole-v1` (seed = 42) |
| `mini_project_episodes.csv` | `python src/bai36.py` | reward và length của 500 episode trong mini-project (Bài 36) |

Định dạng của cả hai file:

```csv
episode,reward,length
1,29.0,29
2,17.0,17
...
```

Các file này được tạo lại mỗi lần chạy script tương ứng. Vì mọi thí nghiệm đều
được seed (`seed = 42`) nên chạy lại sẽ cho đúng cùng một kết quả.
