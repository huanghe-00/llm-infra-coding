"""
【Q05】实现 MQA（Multi-Query Attention）

要求：
1. 与 MHA 区别：K/V 投影后只有 1 个 head（num_kv_heads=1），Q 保持 num_heads
2. 实现 MQA 类，forward(x, past_kv=None)
3. K/V 形状: (batch, 1, seq, head_dim)，Q 形状: (batch, num_heads, seq, head_dim)
4. 计算 attention 时，K/V 需要广播到 num_heads 维度

C++ 程序员注意：
- numpy 广播规则：形状 (1, 1, seq, dim) 可以与 (batch, num_heads, seq, dim) 直接运算
- np.expand_dims 或 reshape 可用于增加维度
"""

import numpy as np
import pytest


class MultiQueryAttention:
    def __init__(self, dim: int, num_heads: int):
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        np.random.seed(42)
        self.w_q = np.random.randn(dim, dim).astype(np.float32) * 0.02
        self.w_k = np.random.randn(dim, self.head_dim).astype(np.float32) * 0.02
        self.w_v = np.random.randn(dim, self.head_dim).astype(np.float32) * 0.02
        self.w_o = np.random.randn(dim, dim).astype(np.float32) * 0.02

    def _softmax(self, x: np.ndarray):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray, past_kv=None):
        # TODO: 实现 MQA 前向
        # 1. Q 投影 reshape -> (batch, num_heads, seq, head_dim)
        # 2. K/V 投影 reshape -> (batch, 1, seq, head_dim)
        # 3. 拼接 past_kv
        # 4. Attention: Q @ K.T，K 广播到 num_heads
        # 5. Causal Mask, Softmax, @ V，V 广播
        # 6. 输出投影
        pass


class TestMQA:
    def test_kv_head_count(self):
        mqa = MultiQueryAttention(dim=64, num_heads=4)
        x = np.random.randn(2, 5, 64).astype(np.float32)
        out, (k, v) = mqa.forward(x, None)
        assert k.shape == (2, 1, 5, 16)  # 只有 1 个 kv head
        assert v.shape == (2, 1, 5, 16)

    def test_output_shape(self):
        mqa = MultiQueryAttention(dim=64, num_heads=8)
        x = np.random.randn(1, 3, 64).astype(np.float32)
        out, _ = mqa.forward(x, None)
        assert out.shape == (1, 3, 64)

    def test_with_pastkv(self):
        mqa = MultiQueryAttention(dim=64, num_heads=4)
        x1 = np.random.randn(2, 3, 64).astype(np.float32)
        out1, (k1, v1) = mqa.forward(x1, None)

        x2 = np.random.randn(2, 1, 64).astype(np.float32)
        out2, (k2, v2) = mqa.forward(x2, (k1, v1))
        assert k2.shape == (2, 1, 4, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
