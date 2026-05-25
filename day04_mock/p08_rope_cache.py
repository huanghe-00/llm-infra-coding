"""
【P08】RoPE 缓存综合题

要求：同 Q03，但增加：
1. 预计算 sin/cos 缓存到 max_seq_len
2. 支持传入 seq_len > max_seq_len 时自动扩展缓存
3. 提供 get_cache(seq_len) 返回 (sin, cos)
"""

import numpy as np
import pytest


class CachedRoPE:
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        # TODO
        pass

    def get_cache(self, seq_len: int):
        # TODO: 如果 seq_len > 当前缓存长度，动态扩展
        pass

    def apply(self, q: np.ndarray, k: np.ndarray, seq_len: int):
        # TODO: 获取 cache 并应用
        pass


class TestCachedRoPE:
    def test_dynamic_extend(self):
        rope = CachedRoPE(head_dim=64, max_seq_len=10)
        sin, cos = rope.get_cache(100)
        assert sin.shape[0] >= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
