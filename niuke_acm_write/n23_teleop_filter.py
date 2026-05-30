"""
【N23】遥操作数据清洗（数据飞轮环节）

【描述】
自变量通过主从遥操采集真机数据，采集后需清洗过滤：
给定 n 条轨迹，每条轨迹包含：
- success: 是否成功（0/1）
- duration_ms: 执行时长（毫秒）
- max_joint_speed: 最大关节速度（rad/s）

过滤规则（全部满足才保留）：
1. success == 1
2. duration_ms <= max_duration（给定上限）
3. max_joint_speed <= max_speed（给定上限，防止危险动作）

输出保留的轨迹数量。

【输入】
第一行：n max_duration_ms max_speed
接下来 n 行：success duration_ms max_joint_speed

【输出】
一个整数，保留轨迹数

【样例输入】
5 5000 2.0
1 3000 1.5
0 4000 1.0
1 6000 1.2
1 2000 2.5
1 1000 0.5

【样例输出】
2

【解释】
保留第1条（成功、3000<=5000、1.5<=2.0）和第5条（成功、1000<=5000、0.5<=2.0）
第2条失败，第3条超时，第4条超速。

【边界】
n=0 输出 0；max_duration 或 max_speed 为 0 时只有恰好等于的才保留；
success 不为 0/1 时视为失败。
"""

import sys

def solve():
    line = sys.stdin.readline().strip()
    n, duration_ms, max_joint_speed = line.split()
    n = int(n)
    duration_ms = int(duration_ms)
    max_joint_speed = float(max_joint_speed)

    valid_num: int = 0
    for _ in range(n):
        line = sys.stdin.readline().strip()
        x, time, speed = line.split()
        x = int(x)
        time = int(time)
        speed = float(speed)
        if x == 1 and time <= duration_ms and speed <= max_joint_speed:
            valid_num += 1
    
    print(valid_num)

if __name__ == "__main__":
    solve()

# import sys


# def solve():
#     line = sys.stdin.readline().strip()
#     if not line:
#         return
#     n, max_dur, max_spd = line.split()
#     n = int(n)
#     max_dur = float(max_dur)
#     max_spd = float(max_spd)

#     kept = 0
#     for _ in range(n):
#         parts = sys.stdin.readline().strip().split()
#         if len(parts) < 3:
#             continue
#         suc = int(parts[0])
#         dur = float(parts[1])
#         spd = float(parts[2])
#         if suc == 1 and dur <= max_dur and spd <= max_spd:
#             kept += 1

#     print(kept)


# if __name__ == "__main__":
#     solve()
