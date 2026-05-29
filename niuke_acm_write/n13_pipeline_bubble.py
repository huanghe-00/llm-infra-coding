"""
【N13】流水线气泡率计算

【描述】
给定流水线 stages 数和 micro_batches 数，计算 bubble rate。
公式：
- total_time = stages + micro_batches - 1
- ideal_time = micro_batches
- bubble_time = total_time - ideal_time = stages - 1
- bubble_rate = bubble_time / total_time

【输入】
一行两个整数：stages micro_batches

【输出】
bubble_rate（保留4位小数）

【样例输入】
4 8

【样例输出】
0.2727

【边界】
stages >= 1, micro_batches >= 1；stages=1 时 bubble_rate=0.0000。
"""
import sys


def solve():
    line = sys.stdin.readline().strip()
    if not line:
        return
    stages, micro = map(int, line.split())
    total = stages + micro - 1
    bubble = stages - 1
    rate = bubble / total if total > 0 else 0.0
    print(f"{rate:.4f}")


if __name__ == "__main__":
    solve()
