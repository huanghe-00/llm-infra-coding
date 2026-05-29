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
    _DTYPE_SIZE: dict[str, int] = {"fp16": 2, "int8" : 1}
    
    def __init__(self, num_layers: int, 
                 num_q_heads: int, 
                 num_kv_heads: int, 
                 head_dim: int, 
                 max_seq_len: int = 2048, 
                 dtype: str = "fp16") -> None:
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        # GQA分组
        self.num_group = num_q_heads // num_kv_heads

        # 预制kv cache
        # 类型注解 list[np.ndarray | None]：告诉类型检查器，这个属性是一个列表，列表里的每个元素要么是 np.ndarray（K 缓存的张量），要么是 None（还没分配）。
        self.k_cache: list[np.ndarray | None] = [None] * num_layers
        self.v_cache: list[np.ndarray | None] = [None] * num_layers

    def append(self, layer_idx: int, new_k: np.ndarray, new_v: np.ndarray) -> None:
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = new_k
            self.v_cache[layer_idx] = new_v
        else:
            self.k_cache[layer_idx] = np.concatenate([self.k_cache[layer_idx], new_k], axis = 2)
            self.v_cache[layer_idx] = np.concatenate([self.v_cache[layer_idx], new_v], axis = 2)

    def get(self, layer_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        # 在Init时，已经声明要么是None，要么是KV缓存，因此这里可以直接返回
        return (self.k_cache[layer_idx], self.v_cache[layer_idx])

    def memory_size(self, batch_size: int = 1) -> int:
        # 计算当前缓存kv_cache的字节数
        item_size = self._DTYPE_SIZE.get(self.dtype, 2)
        total_elems = 0
        for k_layer in self.k_cache:
            if k_layer is not None:
                seq_len = k_layer.shape[2]
                total_elems +=  seq_len
        # total_seq_len * FP16/INT8 * 2(kv) * batchSize * num_kv_head * head_dim
        return total_elems * item_size * 2 * batch_size * self.num_kv_heads * self.head_dim # 提取公共乘法因子，提升计算效率

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
