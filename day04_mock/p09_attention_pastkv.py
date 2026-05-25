"""
【P09】Attention + past_kv 综合题（支持 GQA）

要求：结合 Q02 和 Q05，实现支持 past_kv 的 GQA Attention。
Q 有 num_heads，K/V 有 num_kv_heads，通过广播计算 attention。
"""

import numpy as np
import pytest


class GQAAttention:
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        assert dim % num_heads == 0
        # TODO
        pass

    def forward(self, x: np.ndarray, past_kv=None):
        # TODO
        pass


class TestGQAAttention:
    def test_gqa_pastkv(self):
        attn = GQAAttention(dim=64, num_heads=8, num_kv_heads=2)
        x1 = np.random.randn(1, 2, 64).astype(np.float32)
        out1, (k1, v1) = attn.forward(x1, None)

        x2 = np.random.randn(1, 1, 64).astype(np.float32)
        out2, (k2, v2) = attn.forward(x2, (k1, v1))

        assert k2.shape == (1, 2, 3, 8)  # head_dim=8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
