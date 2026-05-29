"""
【N10】Causal Mask 生成

【描述】
给定序列长度 seq_len，生成下三角 Causal Mask（含对角线）。
Mask[i][j] = 1 如果 j <= i，否则为 0。

【输入】
一个整数 seq_len

【输出】
seq_len 行，每行 seq_len 个整数（0 或 1），空格分隔。

【样例输入】
4

【样例输出】
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1

【边界】
seq_len >= 1；注意 Python 的 range 和列表推导式。
"""
import sys


def solve():
    line = sys.stdin.readline().strip()
    if not line:
        return
    n = int(line)

    for i in range(n):
        row = []
        for j in range(n):
            row.append("1" if j <= i else "0")
        print(" ".join(row))


if __name__ == "__main__":
    solve()
