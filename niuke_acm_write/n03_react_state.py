"""
【N03】ReAct Agent 状态转换

【描述】
给定当前状态和事件，返回下一状态。
状态机：
IDLE + start -> THINK
THINK + act -> ACT
ACT + success -> VERIFY
ACT + failure -> THINK   （重试，但本题不记重试次数）
VERIFY + verify_ok -> DONE
VERIFY + verify_fail -> ACT
任意状态 + timeout -> ERROR
其他未定义的组合 -> 保持当前状态

【输入】
第一行：当前状态字符串（IDLE/THINK/ACT/VERIFY）
第二行：事件字符串（start/act/success/failure/verify_ok/verify_fail/timeout）

【输出】
下一状态字符串。

【样例输入】
THINK
act

【样例输出】
ACT

【边界】
输入可能包含首尾空格，需 strip；
未定义事件时输出当前状态（不是 ERROR）。
"""
import sys


def solve():
    state = sys.stdin.readline().strip()
    event = sys.stdin.readline().strip()

    # 状态转换表：用字典或 if/elif
    if state == "IDLE" and event == "start":
        print("THINK")
    elif state == "THINK" and event == "act":
        print("ACT")
    elif state == "ACT" and event == "success":
        print("VERIFY")
    elif state == "ACT" and event == "failure":
        print("THINK")
    elif state == "VERIFY" and event == "verify_ok":
        print("DONE")
    elif state == "VERIFY" and event == "verify_fail":
        print("ACT")
    elif event == "timeout":
        print("ERROR")
    else:
        print(state)


if __name__ == "__main__":
    solve()
