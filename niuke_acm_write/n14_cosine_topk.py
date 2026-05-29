"""
【N14】余弦相似度 TopK 检索

【描述】
给定 query 向量（dim 维）和 n 个候选向量，计算余弦相似度，返回相似度最高的 k 个候选索引。
余弦相似度：cos(a,b) = (a·b) / (|a||b|)
若 query 向量模长为0，所有相似度视为 -1（不参与排序，除非全部候选也为-1）。
若候选向量模长为0，相似度视为 -1。
按相似度降序，同相似度按索引升序（稳定）。

【输入】
第一行：dim k
第二行：query 向量（dim 个浮点数）
第三行：n（候选数）
接下来 n 行：每行 dim 个浮点数

【输出】
一行 k 个整数（空格分隔），表示 TopK 候选索引。
k=0 时输出空行。

【样例输入】
3 2
1.0 0.0 0.0
3
1.0 0.0 0.0
0.0 1.0 0.0
0.5 0.5 0.0

【样例输出】
0 2

【边界】
k=0 输出空行；k>n 返回全部（按序）；query 全零时返回前 k 个索引。
"""
import sys
import math


def solve():
    line = sys.stdin.readline().strip()
    if not line:
        return
    dim, k = map(int, line.split())
    
    query = list(map(float, sys.stdin.readline().strip().split()))
    n = int(sys.stdin.readline().strip())
    
    q_norm = math.sqrt(sum(x * x for x in query))
    
    sims = []
    for idx in range(n):
        cand = list(map(float, sys.stdin.readline().strip().split()))
        c_norm = math.sqrt(sum(x * x for x in cand))
        if q_norm == 0 or c_norm == 0:
            sim = -1.0
        else:
            dot = sum(a * b for a, b in zip(query, cand))
            sim = dot / (q_norm * c_norm)
        sims.append((-sim, idx))  # 负号用于升序模拟降序
    
    sims.sort(key=lambda x: (x[0], x[1]))
    k = min(k, n)
    res = [str(sims[i][1]) for i in range(k)]
    print(" ".join(res)) if res else print()


if __name__ == "__main__":
    solve()