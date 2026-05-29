
"""
【N01】KV Cache 内存计算（MB）

【描述】
给定 batch, seq_len, num_layers, num_kv_heads, head_dim, bits，
计算 KV Cache 占用内存（MB）。
公式：2 * B * S * L * H * D * bits / 8 / 1024 / 1024

【输入】
一行 6 个整数，空格分隔：
batch seq_len num_layers num_kv_heads head_dim bits

【输出】
一个浮点数（保留 2 位小数），表示内存大小（MB）。

【样例输入】
1 2048 32 8 128 16

【样例输出】
256.00

【边界】
bits 可能为 8/16/32；所有参数 >=1；结果用 float 避免整除截断。
"""

import sys

def solve():
    line = sys.stdin.readline().strip()
    if not line:
        return
    batch, seq_len, num_layers, num_kv_heads, head_dim, bits = map(int, line.split())
    

    # kv_cache的memory size计算公式
    # n bytes = batch * num_layers * 2kv * seq_len * num_kv_heads * head_dim * (bits / 8)
    kv_cache_memory_size = batch * num_layers * 2.0 * seq_len * num_kv_heads * head_dim * bits / 8 / 1024 / 1024
    print(f"{kv_cache_memory_size:.2f}")

if __name__ == "__main__":
    solve()

