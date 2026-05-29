"""
【N20】数据加载理论吞吐量

【描述】
给定 batch_size、num_workers、单样本加载时间（ms）。
假设 num_workers 并行加载，每个 worker 一次加载 batch_size / num_workers 个样本（均分）。
理论每步时间 = single_load_ms * ceil(batch_size / num_workers)
吞吐量 throughput = 1000 / 每步时间 * batch_size（samples/sec）

【输入】
一行3个整数：batch_size num_workers single_load_ms

【输出】
理论吞吐量（保留2位小数）

【样例输入】
4 2 100

【样例输出】
20.00

【解释】
每步时间 = 100 * ceil(4/2) = 200ms；throughput = 1000/200 * 4 = 20 samples/sec

【边界】
num_workers=0 时视为串行（workers=1）；single_load_ms=0 输出 0.00。
"""
import sys
import math


def solve():
    line = sys.stdin.readline().strip()
    if not line:
        return
    batch, workers, load_ms = map(int, line.split())

    if load_ms == 0:
        print("0.00")
        return
    if workers <= 0:
        workers = 1

    per_step = load_ms * math.ceil(batch / workers)
    throughput = 1000.0 / per_step * batch
    print(f"{throughput:.2f}")


if __name__ == "__main__":
    solve()
