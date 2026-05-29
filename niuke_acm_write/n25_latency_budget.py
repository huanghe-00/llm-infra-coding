"""
【N25】实时控制延迟预算分解

【描述】
自变量 WALL-A 模型部署在机器人上，需要满足实时控制要求（总延迟 < budget_ms）。
给定各环节延迟（毫秒，浮点数）：
- vision_encode: 视觉编码
- llm_forward: LLM 推理
- action_gen: 动作生成
- control_send: 控制指令下发
- comm_round: 通信往返

计算总延迟，判断是否满足要求。
若满足，输出 OK 总延迟（保留2位小数）。
若不满足，输出 FAIL 总延迟 瓶颈环节名称（延迟最大的环节）。

【输入】
第一行：budget_ms（浮点数）
第二行：vision_encode llm_forward action_gen control_send comm_round（5个浮点数）

【输出】
OK 总延迟  或  FAIL 总延迟 瓶颈环节

【样例输入】
100.0
20.0 50.0 10.0 5.0 15.0

【样例输出】
OK 100.00

【边界】
总延迟恰好等于 budget 时视为 OK；
多环节同为最大延迟时，按输入顺序取第一个；
延迟为负数时取绝对值（容错）。
"""
import sys


def solve():
    budget = float(sys.stdin.readline().strip())
    parts = sys.stdin.readline().strip().split()
    names = ["vision_encode", "llm_forward", "action_gen", "control_send", "comm_round"]

    delays = [abs(float(p)) for p in parts]
    total = sum(delays)
    max_delay = max(delays)
    bottleneck = names[delays.index(max_delay)]

    if total <= budget:
        print(f"OK {total:.2f}")
    else:
        print(f"FAIL {total:.2f} {bottleneck}")


if __name__ == "__main__":
    solve()
