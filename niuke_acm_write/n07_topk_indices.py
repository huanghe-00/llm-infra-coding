"""
【N07】Top-K 索引选择

【描述】
给定 n 个浮点数（logits），返回值最大的 k 个元素的索引（按值从高到低排序）。
如果 k > n，返回所有索引。

【输入】
第一行：n k（整数）
第二行：n 个浮点数（logits）

【输出】
一行 k 个整数（空格分隔），表示 Top-K 索引。

【样例输入】
5 3
1.0 3.0 2.0 5.0 0.5

【样例输出】
3 1 2

【边界】
k 可能为 0（输出空行）；n 可能为 1；logits 可能有负数；值相同时索引小的优先（稳定排序）。
"""
# lamdba x: (-x[0], x[1])
# 等价于
# def 匿名函数:
#     return (-x[0], x[1])

import sys

def solve():
    try:
        n, k = map(int, sys.stdin.readline().strip().split())
        logits: List[float] = list(map(float, sys.stdin.readline().strip().split()))
        
        indexed_logits = [(idx, logit) for idx, logit in enumerate(logits)]
        indexed_logits.sort(key=lambda x: (-x[1], x[0]))  # 首位按-logit升序排序，次位按idx升序排序
        print(" ".join(f"{logit:.2f}" for _, logit in indexed_logits), file=sys.stderr)
        print(" ".join(f"{idx}" for idx, _ in indexed_logits[0:min(k, n)]))

    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    solve()

# import sys


# def solve():
#     line = sys.stdin.readline().strip()
#     if not line:
#         return
#     n, k = map(int, line.split())
#     logits = list(map(float, sys.stdin.readline().strip().split()))

#     # 带索引排序，按值降序，值相同按索引升序（稳定）
#     indexed = [(val, idx) for idx, val in enumerate(logits)]
#     indexed.sort(key=lambda x: (-x[0], x[1]))

#     k = min(k, n)
#     topk = [str(indexed[i][1]) for i in range(k)]
#     print(" ".join(topk)) if topk else print()


# if __name__ == "__main__":
#     solve()
