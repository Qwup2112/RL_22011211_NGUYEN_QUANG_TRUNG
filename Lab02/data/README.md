# Thư mục `data/`

Nơi lưu dữ liệu thí nghiệm sinh ra khi chạy code trong `Lab02/src/`.

| File | Sinh ra bởi | Nội dung |
|------|-------------|----------|
| `value_iteration_deltas.csv` | `python src/bai36.py` | chuỗi `delta = max\|V_mới − V_cũ\|` của Value Iteration ở mỗi iteration, cho cả hai cấu hình `is_slippery=False/True` |

Định dạng:

```csv
setting,iteration,delta
is_slippery=False,1,1.0
is_slippery=False,2,0.99
...
is_slippery=True,438,9.77e-09
```

Dùng để vẽ lại đường hội tụ mà không phải chạy lại thuật toán. Mọi thí nghiệm
đều có seed (`seed = 42`) và thuật toán DP là tất định, nên chạy lại cho đúng
cùng một kết quả.
