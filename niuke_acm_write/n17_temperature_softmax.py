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
from typing import List

def solve():
    try:
        line = sys.stdin.readline().strip()
        if not line:
            return
        n = int(line.split()[0])
        temperature = float(line.split()[1])
        logits: List[float] = list(map(float, sys.stdin.readline().strip().split()))
        assert n == len(logits), f"输入数据不匹配"
        # 这里需要校验t的合法性，不可以小于0，不过题目好像没要求，省略
        results: List[float] = [0.0 for _ in range(n)]
        if abs(temperature) < 1e-9:
            idx_max = -1
            logit_max = -1e9
            for i, logit in enumerate(logits):
                if logit > logit_max:
                    logit_max = logit
                    idx_max = i
            results[idx_max] = 1.0
            print(" ".join(f"{result:.4f}" for result in results))
            return
        
        # 为logits计算softmax
        scaled = [logit / temperature for logit in logits]
        scaled_max = max(scaled)
        total_e = 0.0
        for i, _ in enumerate(scaled):
            scaled[i] = (scaled[i] - scaled_max)
            scaled[i] = math.exp(scaled[i])
            total_e += scaled[i]
        for i, _ in enumerate(scaled):
            scaled[i] /= total_e
        print(" ".join(f"{x:.4f}" for x in scaled))

    except Exception:
        import traceback
        trackback.print_exc()

if __name__ == "__main__":
    solve()


# import sys
# import math



# def solve():
#     line = sys.stdin.readline().strip()
#     if not line:
#         return
#     n, temp = line.split()
#     n = int(n)
#     temp = float(temp)

#     logits = list(map(float, sys.stdin.readline().strip().split()))

#     if temp == 0:
#         mx = max(logits)
#         res = []
#         found = False
#         for v in logits:
#             if not found and v == mx:
#                 res.append("1.0000")
#                 found = True
#             else:
#                 res.append("0.0000")
#         print(" ".join(res))
#         return

#     scaled = [v / temp for v in logits]
#     mx = max(scaled)
#     exps = [math.exp(v - mx) for v in scaled]
#     s = sum(exps)
#     probs = [f"{e / s:.4f}" for e in exps]
#     print(" ".join(probs))


# if __name__ == "__main__":
#     solve()
