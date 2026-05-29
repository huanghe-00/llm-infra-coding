"""
【N02】动作 Tokenization（连续值 -> 离散 bin）

【描述】
将机械臂的连续动作值编码为离散 Token。
步骤：
1. clip：val = max(min_val, min(val, max_val))
2. bin_size = (max_val - min_val) / bins
3. idx = int((val - min_val) / bin_size)
4. idx = min(idx, bins - 1)  # 边界保护
5. 返回 idx

【输入】
一行 4 个值，空格分隔：
action_val(float) min_val(float) max_val(float) bins(int)

【输出】
一个整数，表示离散 Token ID。

【样例输入】
0.5 -1.0 1.0 256

【样例输出】
192

【边界】
action_val 可能越界（如 1.5 或 -2.0），必须先 clip；
bins 为 256 时，idx 必须在 [0, 255]。
"""

import sys

def solve:
    line = sys.stdin.readline().strip()
    if not line:
        return
    parts = line.split()
    action_val, min_val, max_val = map(float, parts[:3])
    bins = int(parts[3])
    







import sys


def solve():
    line = sys.stdin.readline().strip()
    if not line:
        return
    parts = line.split()
    val, min_v, max_v = map(float, parts[:3])
    bins = int(parts[3])

    # TODO: clip -> bin -> int -> boundary protect
    clipped = max(min_v, min(val, max_v))
    if max_v == min_v:
        print(0)
        return
    bin_size = (max_v - min_v) / bins
    idx = int((clipped - min_v) / bin_size)
    idx = min(idx, bins - 1)
    print(idx)


if __name__ == "__main__":
    solve()
