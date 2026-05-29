"""
【N12】连续批处理一步模拟

【描述】
给定当前 batch 中各请求的剩余 token 数（空格分隔，0 表示已完成），
以及等待队列中各请求的 prompt_len（空格分隔）。
模拟一次 step()：
1. 当前 batch 中所有请求 remaining -= 1
2. 移除 remaining <= 0 的请求
3. 从等待队列头部补充新请求，直到 batch 满（保持 batch 最大容量不变）
4. 新加入的请求在加入后立即 -1（即参与本次 step 的消耗）
输出一步后 batch 中所有请求的剩余 token 数（空格分隔，按请求顺序）。

【输入】
第一行：max_batch_size
第二行：当前 batch 剩余 tokens（空格分隔，可能为空行表示 batch 空）
第三行：等待队列 prompt_lens（空格分隔，可能为空行）

【输出】
一行，空格分隔的整数（剩余 tokens），按请求在 batch 中的顺序。
若 batch 为空，输出空行。

【样例输入】
2
1 3
2

【样例输出】
2 1

【解释】
原 batch: [1,3] -> step后 [0,2] -> 移除0 -> [2]
等待队列: [2] -> 加入 -> [2,2] -> 新加入的-1 -> [2,1]

【边界】
输入可能含空行；batch 和等待队列都可能为空。
"""
import sys


def solve():
    max_batch = int(sys.stdin.readline().strip())

    line2 = sys.stdin.readline().strip()
    batch = list(map(int, line2.split())) if line2 else []

    line3 = sys.stdin.readline().strip()
    waiting = list(map(int, line3.split())) if line3 else []

    # Step 1: batch内全部-1
    batch = [x - 1 for x in batch]
    # Step 2: 移除<=0的
    batch = [x for x in batch if x > 0]

    # Step 3: 从waiting补充，直到batch满
    while len(batch) < max_batch and waiting:
        new_req = waiting.pop(0)
        batch.append(new_req - 1)  # 新加入的立即-1

    if batch:
        print(" ".join(map(str, batch)))
    else:
        print()


if __name__ == "__main__":
    solve()
