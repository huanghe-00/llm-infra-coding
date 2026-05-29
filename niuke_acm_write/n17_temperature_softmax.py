"""
【N17】Temperature Softmax

【描述】
给定 n 个 logits 和一个 temperature，计算 temperature-scaled softmax。
步骤：
1. logits / temperature
2. 减去最大值（数值稳定）
3. exp -> 归一化

【输入】
第一行：n temperature（temperature 为浮点数，temperature=0 时视为 greedy，直接输出 argmax 处为1，其余为0）
第二行：n 个浮点数（logits）

【输出】
n 个概率（保留4位小数，空格分隔）

【样例输入】
3 1.0
1.0 2.0 3.0

【样例输出】
0.0900 0.2447 0.6652

【边界】
temperature=0 时 greedy；极大负数 logtis；同值 logits。
"""
import sys
import math


def solve():
    line = sys.stdin.readline().strip()
    if not line:
        return
    n, temp = line.split()
    n = int(n)
    temp = float(temp)

    logits = list(map(float, sys.stdin.readline().strip().split()))

    if temp == 0:
        mx = max(logits)
        res = []
        found = False
        for v in logits:
            if not found and v == mx:
                res.append("1.0000")
                found = True
            else:
                res.append("0.0000")
        print(" ".join(res))
        return

    scaled = [v / temp for v in logits]
    mx = max(scaled)
    exps = [math.exp(v - mx) for v in scaled]
    s = sum(exps)
    probs = [f"{e / s:.4f}" for e in exps]
    print(" ".join(probs))


if __name__ == "__main__":
    solve()
