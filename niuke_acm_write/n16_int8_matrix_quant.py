"""
【N16】INT8 矩阵量化

【描述】
给定一个 n x m 的浮点矩阵，以及量化方式：
- 若输入为 auto：自动计算 scale = max(abs(matrix)) / 127.0，然后逐元素 round(val / scale)
- 若输入为具体浮点数：使用该值作为 scale
输出量化后的 INT8 矩阵（n 行，每行 m 个整数，空格分隔）。
若 max_abs=0 且 auto，则全输出 0。

【输入】
第一行：n m
接下来 n 行：每行 m 个浮点数
最后一行：scale_str（auto 或浮点数）

【输出】
n 行，每行 m 个整数（空格分隔）

【样例输入】
2 2
1.0 -1.0
0.5 -0.5
auto

【样例输出】
127 -127
64 -64

【边界】
全0矩阵；单元素；给定scale与auto结果可能不同。
"""
import sys


def solve():
    line = sys.stdin.readline().strip()
    if not line:
        return
    n, m = map(int, line.split())

    mat = []
    for _ in range(n):
        row = list(map(float, sys.stdin.readline().strip().split()))
        mat.append(row)

    scale_str = sys.stdin.readline().strip()

    flat = [abs(x) for row in mat for x in row]
    max_abs = max(flat) if flat else 0.0

    if scale_str == "auto":
        scale = max_abs / 127.0 if max_abs > 0 else 0.0
    else:
        scale = float(scale_str)

    for row in mat:
        if scale == 0:
            print(" ".join("0" for _ in row))
        else:
            q = [str(int(round(x / scale))) for x in row]
            print(" ".join(q))


if __name__ == "__main__":
    solve()
