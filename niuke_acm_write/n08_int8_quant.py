"""
【N08】INT8 对称量化与反量化

【描述】
给定 n 个浮点数，进行 INT8 对称量化：
scale = max(abs(x)) / 127.0
q = round(x / scale)
然后反量化：x_hat = q * scale
输出 scale，以及反量化后的 n 个值（保留 4 位小数）。

【输入】
第一行：n
第二行：n 个浮点数

【输出】
第一行：scale（保留 6 位小数）
第二行：n 个反量化值（保留 4 位小数）

【样例输入】
3
-1.5 0.0 1.5

【样例输出】
0.011811
-1.5000 0.0000 1.5000

【边界】
n 可能为 1；所有输入为 0 时 scale 理论上为 0，此时输出 0 并把反量化值全输出 0；
注意 round() 返回 float，需转为 int 再乘 scale。
"""
import sys
from typing import List

def solve():
    try:
        n = int(sys.stdin.readline().strip())
        nums_for_quant: List[float] = list(map(float, sys.stdin.readline().strip().split()))
        nums_max = -1e9
        for num in nums_for_quant:
            if abs(num) > nums_max:
                nums_max = abs(num)
        scale = nums_max / 127.0
        if scale == 0.0:
            print(f"{scale:.6f}")
            print(" ".join("0.0000" for _ in nums_for_quant))
            return

        print(f"{scale:.6f}")
        quant = [int(round(num / scale))  for num in nums_for_quant]
        re_quant = [float(q * scale)  for q in quant]
        print(" ".join(f"{rq:.4f}" for rq in re_quant))

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
#     n = int(line)
#     arr = list(map(float, sys.stdin.readline().strip().split()))

#     max_abs = max(abs(x) for x in arr)
#     if max_abs == 0:
#         print("0.000000")
#         print(" ".join("0.0000" for _ in arr))
#         return

#     scale = max_abs / 127.0
#     dequant = [round(x / scale) * scale for x in arr]

#     print(f"{scale:.6f}")
#     print(" ".join(f"{x:.4f}" for x in dequant))


# if __name__ == "__main__":
#     solve()
