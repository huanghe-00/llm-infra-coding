"""
【N18】重试计数器

【描述】
给定最大重试次数 max_retries，以及一个事件序列（空格分隔）。
从 IDLE 状态开始处理事件：
- success -> 立即 SUCCESS
- failure -> 重试计数+1，若超过 max_retries 则 FAILED
- timeout -> 立即 ERROR
- 空序列 -> SUCCESS（无操作视为成功）
输出最终状态（SUCCESS/FAILED/ERROR）和实际重试次数（整数，空格分隔）。

【输入】
第一行：max_retries
第二行：事件序列（空格分隔，可能为空行）

【输出】
最终状态 重试次数

【样例输入】
2
failure failure success

【样例输出】
SUCCESS 2

【边界】
连续 failure 刚好达到上限；timeout 优先于重试计数；空序列。
"""
import sys


def solve():
    max_retries = int(sys.stdin.readline().strip())
    line = sys.stdin.readline().strip()
    events = line.split() if line else []

    retries = 0
    for e in events:
        if e == "success":
            print(f"SUCCESS {retries}")
            return
        elif e == "timeout":
            print(f"ERROR {retries}")
            return
        elif e == "failure":
            retries += 1
            if retries > max_retries:
                print(f"FAILED {max_retries}")
                return

    print(f"SUCCESS {retries}")


if __name__ == "__main__":
    solve()
