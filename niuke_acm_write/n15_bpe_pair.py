"""
【N15】BPE 相邻字符对统计

【描述】
给定一个无空格的字符串，统计所有相邻字符对（长度为2的子串）的出现频次。
返回出现频次最高的字符对及其频次。若有多个同频次，返回字典序最小的。
若无任何相邻对（字符串长度<2），输出 NONE。

【输入】
一行字符串（仅含小写字母，无空格）

【输出】
字符对 频次（空格分隔），或 NONE

【样例输入】
aabbc

【样例输出】
aa 1

【解释】
aa出现1次，ab出现1次，bb出现1次，bc出现1次。同频次取字典序最小aa。

【边界】
长度<2输出NONE；全相同字符如aaaa -> aa 3。
"""
import sys
from collections import Counter


def solve():
    s = sys.stdin.readline().strip()
    if len(s) < 2:
        print("NONE")
        return

    pairs = [s[i:i+2] for i in range(len(s) - 1)]
    cnt = Counter(pairs)
    max_freq = max(cnt.values())
    best = min([p for p, c in cnt.items() if c == max_freq])
    print(f"{best} {max_freq}")


if __name__ == "__main__":
    solve()
