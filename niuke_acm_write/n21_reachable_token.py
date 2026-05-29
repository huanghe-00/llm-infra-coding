"""
【N21】具身动作空间可达性判断 + Tokenization

【描述】
自变量 WALL-A 模型输出末端执行器目标位姿 (x,y,z,rx,ry,rz)，
需先判断是否在机械臂可达工作空间内，再编码为动作 Token。

工作空间判定（简化球模型）：
- 机械臂基座在原点 (0,0,0)
- 可达空间为半径 R 的球，且 z >= 0（不能钻地）
- 即：sqrt(x^2+y^2+z^2) <= R 且 z >= 0

若不可达，输出 UNREACHABLE。
若可达，将 6 个维度分别 clip 到 [-1,1]，再按 bins=256 均匀离散化，
输出 6 个整数 Token（空格分隔）。

离散化公式：idx = int((v + 1.0) / 2.0 * 256)
然后 clip 到 [0, 255]。

【输入】
第一行：R（工作空间半径，浮点数）
第二行：x y z rx ry rz（6 个浮点数，目标位姿）

【输出】
UNREACHABLE 或 6 个整数 Token

【样例输入】
1.5
0.5 0.5 0.5 0.0 0.0 0.0

【样例输出】
192 192 192 128 128 128

【边界】
R <= 0 时所有输入都 UNREACHABLE；z < 0 时 UNREACHABLE；
恰好贴边（距离=R 且 z=0）视为 REACHABLE。
"""
import sys
import math


def solve():
    line = sys.stdin.readline().strip()
    if not line:
        return
    R = float(line)
    x, y, z, rx, ry, rz = map(float, sys.stdin.readline().strip().split())
    
    if R <= 0:
        print("UNREACHABLE")
        return
    
    dist = math.sqrt(x*x + y*y + z*z)
    if dist > R or z < 0:
        print("UNREACHABLE")
        return
    
    # 6 个维度分别 clip 到 [-1,1]，再离散化到 256 bins
    tokens = []
    for val in [x, y, z, rx, ry, rz]:
        clipped = max(-1.0, min(1.0, val))
        # 映射到 [0, 256)，然后 clip 到 [0, 255]
        idx = int((clipped + 1.0) / 2.0 * 256)
        idx = min(255, max(0, idx))
        tokens.append(idx)
    
    print(" ".join(map(str, tokens)))


if __name__ == "__main__":
    solve()