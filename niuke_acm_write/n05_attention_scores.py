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
import sys, math
from typing import List

def solve():
    try:
        n, d = map(int, sys.stdin.readline().strip().split())
        Q: List[List[float]] = []
        K: List[List[float]] = []
        for _ in range(n):
            row_n = list(map(float, sys.stdin.readline().strip().split()))
            Q.append(row_n)
            assert len(row_n) == d, f"维度错误"
        for _ in range(n):
            row_n = list(map(float, sys.stdin.readline().strip().split()))
            K.append(row_n)
            assert len(row_n) == d, f"维度错误"
        

        # 1. 注意力计算 Q @ KT  Q矩阵n*d  KT矩阵 d*n  scores n * n
        scores: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):  # 遍历Q的行
            for j in range(n):  # 遍历K的行
                s = 0.0
                for k in range(d):  # 共同维度 d, Q的第k列 x K的第k列
                    s += Q[i][k] * K[j][k]
                    # print("====111编译验证成功===", file=sys.stderr)
                scores[i][j] = s / math.sqrt(d) # 除以sqrt(d)
                # print("====222编译验证成功===", file=sys.stderr)
        
        # 2. mask掩码 右上角标-1e9 并与scores相加
        scores_mask: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
        mask: List[List[float]] = [[-1e9 if j > i else 0 for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                scores_mask[i][j] = scores[i][j] + mask[i][j]
        
        # 3. 对矩阵【逐行】进行sofrmax
        # weight: List[List[float]] = [[scores_mask[i][j] for i in range(n)] for j in range(n)]
        weight = scores_mask
        for i in range(n):
            # 每行softmax
            score_max:float = -1e9
            score_max = max(weight[i])
            sum_e: float = 0.0
            for j in range(n):
                weight[i][j] = math.exp(weight[i][j] - score_max)
                sum_e += weight[i][j]
            for j in range(n):
                weight[i][j] = weight[i][j] / sum_e
    
        for i in range(n):
            print(" ".join(f"{x:.04f}" for x in weight[i]))
        print("====编译验证成功===", file=sys.stderr)
    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    solve()


# import sys
# import math


# def solve():
#     line = sys.stdin.readline().strip()
#     if not line:
#         return
#     n, d = map(int, line.split())

#     Q = []
#     for _ in range(n):
#         row = list(map(float, sys.stdin.readline().strip().split()))
#         Q.append(row)

#     K = []
#     for _ in range(n):
#         row = list(map(float, sys.stdin.readline().strip().split()))
#         K.append(row)

#     # TODO: 计算 scores = Q @ K.T / sqrt(d)
#     # 注意：K.T 即 K 的转置
#     scores = [[0.0] * n for _ in range(n)]
#     for i in range(n):
#         for j in range(n):
#             s = 0.0
#             for k in range(d):
#                 s += Q[i][k] * K[j][k]
#             scores[i][j] = s / math.sqrt(d)

#     # Causal Mask：上三角（j > i）设为 -1e9
#     for i in range(n):
#         for j in range(n):
#             if j > i:
#                 scores[i][j] = -1e9

#     # Softmax 逐行
#     for i in range(n):
#         row_max = max(scores[i])
#         exps = [math.exp(x - row_max) for x in scores[i]]
#         sum_exp = sum(exps)
#         weights = [x / sum_exp for x in exps]
#         # 输出
#         out = " ".join(f"{w:.4f}" for w in weights)
#         print(out)


# if __name__ == "__main__":
#     solve()
