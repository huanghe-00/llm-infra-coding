"""
【N06】RoPE 位置编码（pair-wise 实现）

【描述】
给定位置 pos 和维度 dim（偶数），计算 RoPE 的频率编码向量。
使用 pair-wise 方式：每对维度 (2i, 2i+1) 共享一个 theta。
theta_i = 10000 ^ (-2*i/dim)，其中 i = 0, 1, ..., dim/2-1
输出第 2i 维：sin(pos * theta_i)
输出第 2i+1 维：cos(pos * theta_i)

【输入】
一行两个整数：pos dim

【输出】
一行 dim 个浮点数（保留 4 位小数）。

【样例输入】
1 4

【样例输出】
0.8415 0.5403 0.0100 1.0000

【边界】
dim 为偶数且 >= 2；pos >= 0。
"""
import sys
import math


def solve():
    line = sys.stdin.readline().strip()
    if not line:
        return
    pos, dim = map(int, line.split())
    
    res = []
    for i in range(0, dim, 2):
        # i 是 pair index，对应维度 2i 和 2i+1
        pair_idx = i // 2
        theta = math.pow(10000.0, -2.0 * pair_idx / dim)
        val_sin = math.sin(pos * theta)
        val_cos = math.cos(pos * theta)
        res.append(f"{val_sin:.4f}")
        res.append(f"{val_cos:.4f}")
    print(" ".join(res))


if __name__ == "__main__":
    solve()