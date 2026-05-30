"""
【N11】MQA/GQA 压缩比计算

【描述】
给定 num_q_heads 和 num_kv_heads，计算压缩比（q_heads / kv_heads）。
要求：
1. 必须能整除（num_q_heads % num_kv_heads == 0）
2. 两者必须都 > 0
3. 否则输出 ERROR

【输入】
一行两个整数：num_q_heads num_kv_heads

【输出】
压缩比（整数），或 ERROR

【样例输入】
32 8

【样例输出】
4

【边界】
不能整除、含0、负数均输出 ERROR。
"""
import sys

def solve():
    num_q_heads, num_kv_heads = map(int, sys.stdin.readline().strip().split())
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        print("ERROR")
        return
    print(f"{int(num_q_heads / num_kv_heads)}")

if __name__ == "__main__":
    solve()

# import sys


# def solve():
#     line = sys.stdin.readline().strip()
#     if not line:
#         return
#     q, kv = map(int, line.split())
#     if q <= 0 or kv <= 0 or q % kv != 0:
#         print("ERROR")
#         return
#     print(q // kv)


# if __name__ == "__main__":
#     solve()
