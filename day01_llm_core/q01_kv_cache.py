"""
【Q01】实现一个支持 GQA 共享的 KV Cache 管理器

要求：
1. 实现 KVCache 类，支持 append(new_k, new_v) 和 get() 方法
2. 支持 GQA：group_size 个 query head 共享 1 个 kv head（本类只管理 kv head，不管理 q head）
3. 当 seq_len 超过 max_seq_len 时，抛出 RuntimeError
4. 提供 memory_size() 方法，返回当前 KV Cache 占用的元素个数

C++ 程序员注意：
- Python 的 list.append() 是引用追加！如果 new_k 是 numpy 数组，
  后续修改 new_k 会污染 cache。但本题中 new_k 是每次新创建的，暂不处理 deepcopy。
- 用 self.cache_k[layer_idx] = [] 来模拟每层一个 vector。
"""

import numpy as np
import pytest


class KVCache:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 max_seq_len: int = 2048):
        # TODO: 初始化
        """
        初始化KV Cache
        参数：
            num_layers
            num_kv_heads
            head_dim
            max_seq_len
        """
        # self.cache_k[layer] 是一个 list，每个元素是 (batch, num_kv_heads, 1, head_dim)
        self.numlayers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        self.cache_k = [[] for _ in range(num_layers)]
        self.cache_v = [[] for _ in range(num_layers)]

        [[] for _ in range(num_layers)]

    def append(self, layer_idx: int, new_k: np.ndarray, new_v: np.ndarray):
        # TODO: 追加 K/V，检查长度是否超过 max_seq_len
        """
        每调用一次 append, 就在该层的列表末尾添加一个形状为 (batch, num_kv_heads, 1, head_dim) 的张量，其中第 2 维长度为 1, 代表“一个时间步”。
        """
        k_cache = self.cache_k[layer_idx]
        if len(k_cache) >= self.max_seq_len:
            raise RuntimeError(f"Error, k_cache len exceeded")
        self.cache_k[layer_idx].append(new_k)
        self.cache_v[layer_idx].append(new_v)

    def get(self, layer_idx: int):
        # TODO: 将 list 沿 seq_len 维度拼接，返回 (k_cache, v_cache)
        # 形状: (batch, num_kv_heads, seq_len, head_dim)
        """
        获取指定曾当前的K和V缓存
        """
        k_list = self.cache_k[layer_idx]
        v_list = self.cache_v[layer_idx]

        if not k_list:
            return (
                np.empty((0, self.num_kv_heads, 0, self.head_dim)),
                np.empty((0, self.num_kv_heads, 0, self.head_dim)),
            )
        # 将所有 token 的片段合并成一个沿序列长度方向连续的大张量
        # 形状从 (batch, heads, 1, dim) 变成 (batch, heads, seq_len, dim)。因此必须沿轴 2（时间步维度）拼接
        k_cache = np.concatenate(k_list, axis = 2)
        v_cache = np.concatenate(v_list, axis = 2)
        return k_cache, v_cache

    def memory_size(self) -> int:
        # TODO: 计算所有层的 K+V 元素总数
        """
        计算KV Cache当前占用的元素总个数
        """
        """
        把多个可迭代对象（如列表）“拉链式”地配对，返回一个迭代器
        a = [1, 2, 3]
        b = ['x', 'y', 'z']
        list(zip(a, b))   # [(1, 'x'), (2, 'y'), (3, 'z')]
        """
        total = 0
        for layer_k, layer_v in zip(self.cache_k, self.cache_v):
            for arr in layer_k:
                total += arr.size
            for arr in layer_v:
                total += arr.size
        return total

class TestKVCache:
    def test_basic_append_and_get(self):
        cache = KVCache(num_layers=2, num_kv_heads=4, head_dim=64)
        for _ in range(3):
            k = np.zeros((1, 4, 1, 64))
            v = np.zeros((1, 4, 1, 64))
            cache.append(0, k, v)

        k_cache, v_cache = cache.get(0)
        assert k_cache.shape == (1, 4, 3, 64)
        assert v_cache.shape == (1, 4, 3, 64)

    def test_gqa_kv_head_count(self):
        """8 个 q head 共享 2 个 kv head, kv head 数保持 2"""
        cache = KVCache(num_layers=1, num_kv_heads=2, head_dim=32)
        k = np.zeros((1, 2, 1, 32))
        v = np.zeros((1, 2, 1, 32))
        cache.append(0, k, v)

        k_cache, _ = cache.get(0)
        assert k_cache.shape[1] == 2

    def test_exceed_max_seq_len(self):
        cache = KVCache(num_layers=1, num_kv_heads=2, head_dim=32, max_seq_len=3)
        for _ in range(3):
            cache.append(0, np.zeros((1, 2, 1, 32)), np.zeros((1, 2, 1, 32)))

        with pytest.raises(RuntimeError):
            cache.append(0, np.zeros((1, 2, 1, 32)), np.zeros((1, 2, 1, 32)))

    def test_memory_size(self):
        cache = KVCache(num_layers=2, num_kv_heads=4, head_dim=64)
        for _ in range(5):
            cache.append(0, np.zeros((1, 4, 1, 64)), np.zeros((1, 4, 1, 64)))

        # layer0: 5 tokens * 4 heads * 64 dim * 2(K+V) = 2560
        assert cache.memory_size() == 2560


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
