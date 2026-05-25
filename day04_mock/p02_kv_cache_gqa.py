"""
【P02】KV Cache 综合题（GQA + 内存计算）

要求：实现 GQAKVCache，在 Q01 基础上增加：
1. 支持 group_size：num_q_heads // num_kv_heads = group_size
2. memory_size() 支持 batch_size 参数，计算总内存（元素数）
3. 支持 FP16/INT8 精度切换（用 dtype 字符串模拟，只计算元素数 * bytes_per_element）

C++ 程序员注意：
- 用 dict 映射 dtype_str -> bytes: {"fp16": 2, "int8": 1}
"""

import numpy as np
import pytest


class GQAKVCache:
    def __init__(self, num_layers: int, num_q_heads: int, num_kv_heads: int,
                 head_dim: int, max_seq_len: int = 2048, dtype: str = "fp16"):
        # TODO
        pass

    def append(self, layer_idx: int, new_k: np.ndarray, new_v: np.ndarray):
        # TODO
        pass

    def get(self, layer_idx: int):
        # TODO
        pass

    def memory_size(self, batch_size: int = 1) -> int:
        # TODO: 总元素数 * bytes_per_element * batch_size
        pass


class TestGQAKVCache:
    def test_gqa_memory(self):
        cache = GQAKVCache(num_layers=2, num_q_heads=8, num_kv_heads=2,
                           head_dim=64, dtype="fp16")
        for _ in range(4):
            cache.append(0, np.zeros((1, 2, 1, 64)), np.zeros((1, 2, 1, 64)))

        # layer0: 4 tokens * 2 kv heads * 64 dim * 2(K+V) = 1024 elements
        # fp16 = 2 bytes, batch=1 -> 2048 bytes
        assert cache.memory_size(batch_size=1) == 2048

    def test_int8_memory(self):
        cache = GQAKVCache(num_layers=1, num_q_heads=4, num_kv_heads=1,
                           head_dim=32, dtype="int8")
        cache.append(0, np.zeros((2, 1, 1, 32)), np.zeros((2, 1, 1, 32)))  # batch=2
        # 1 token * 1 head * 32 * 2(K+V) * 2(batch) = 128 elements
        # int8 = 1 byte -> 128 bytes
        assert cache.memory_size(batch_size=2) == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
