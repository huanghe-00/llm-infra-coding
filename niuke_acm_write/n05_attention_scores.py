"""
【N05】简化 Attention 分数计算（Causal Mask + Softmax）

【描述】
给定 Q 矩阵和 K 矩阵（均为 n x d），计算 causal attention weights。
步骤：
1. scores = Q @ K.T / sqrt(d)
2. Causal Mask：上三角（不含对角线）设为 -inf（用一个极大负数 -1e9 代替）
3. Softmax：逐行计算 exp(x - max) / sum(exp(x - max))

【输入】
第一行：n d（整数，n <= 20, d <= 64）
接下来 n 行：Q 矩阵，每行 d 个浮点数
接下来 n 行：K 矩阵，每行 d 个浮点数

【输出】
n 行，每行 n 个浮点数（保留 4 位小数），表示 softmax 后的 attention weights。

【样例输入】
3 2
1 0
0 1
1 1
1 0
0 1
1 1

【样例输出】
1.0000 0.0000 0.0000
0.0000 1.0000 0.0000
0.5000 0.5000 0.0000

【边界】
n 可能为 1；d 可能为 1；注意除以 sqrt(d) 时 d 为 float。
"""
import sys
import math


def solve():
    line = sys.stdin.readline().strip()
    if not line:
        return
    n, d = map(int, line.split())

    Q = []
    for _ in range(n):
        row = list(map(float, sys.stdin.readline().strip().split()))
        Q.append(row)

    K = []
    for _ in range(n):
        row = list(map(float, sys.stdin.readline().strip().split()))
        K.append(row)

    # TODO: 计算 scores = Q @ K.T / sqrt(d)
    # 注意：K.T 即 K 的转置
    scores = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(d):
                s += Q[i][k] * K[j][k]
            scores[i][j] = s / math.sqrt(d)

    # Causal Mask：上三角（j > i）设为 -1e9
    for i in range(n):
        for j in range(n):
            if j > i:
                scores[i][j] = -1e9

    # Softmax 逐行
    for i in range(n):
        row_max = max(scores[i])
        exps = [math.exp(x - row_max) for x in scores[i]]
        sum_exp = sum(exps)
        weights = [x / sum_exp for x in exps]
        # 输出
        out = " ".join(f"{w:.4f}" for w in weights)
        print(out)


if __name__ == "__main__":
    solve()
