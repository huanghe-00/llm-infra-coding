"""
【N09】优先级请求调度（简化 Batch 组装）

【描述】
给定 n 个请求，每个请求有 req_id（字符串）和 priority（整数，数值越小优先级越高）。
需要选出最多 max_batch 个请求，按优先级从高到低（数值从小到大）排序，
同优先级按输入顺序（稳定排序）。
输出选中的 req_id 列表。

【输入】
第一行：n max_batch
接下来 n 行：req_id priority

【输出】
一行，空格分隔的 req_id（按调度顺序）。

【样例输入】
5 3
req_a 2
req_b 0
req_c 1
req_d 2
req_e 3

【样例输出】
req_b req_c req_a

【边界】
max_batch 可能 >= n（全选）；priority 可能相同；req_id 无空格。
"""
import sys


def solve():
    line = sys.stdin.readline().strip()
    if not line:
        return
    n, max_batch = map(int, line.split())

    reqs = []
    for i in range(n):
        rid, pri = sys.stdin.readline().strip().split()
        reqs.append((int(pri), i, rid))  # (priority, original_index, id)

    # 排序：priority 升序，同优先级按原始索引升序（稳定）
    reqs.sort(key=lambda x: (x[0], x[1]))

    k = min(max_batch, n)
    result = [reqs[i][2] for i in range(k)]
    print(" ".join(result))


if __name__ == "__main__":
    solve()
