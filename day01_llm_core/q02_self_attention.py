"""
【Q02】实现支持 past_kv 的 Causal Self-Attention（NumPy 纯手写版）

要求：
1. 实现 CausalSelfAttention 类，forward(x, past_kv=None) 返回 (output, (new_k, new_v))
2. x 形状: (batch, seq_len, dim)
3. past_kv: (k, v)，形状各为 (batch, num_heads, past_seq, head_dim)，可为 None
4. 计算 Q/K/V 投影（用随机矩阵模拟，不训练）
5. Causal Mask：下三角（含对角线）为 0，上三角为 -inf
6. Softmax 注意力，输出投影

C++ 程序员注意：
- np.matmul 就是矩阵乘法，相当于 C++ 的 A * B
- np.transpose 用于转置维度，相当于 permute
- 注意 past_kv 为 None 时的初始化，不要写 if(past_kv)（Python 中元组恒真），要写 if past_kv is not None
"""

import numpy as np
import pytest


class CausalSelfAttention:
    def __init__(self, dim: int, num_heads: int):
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        np.random.seed(42)
        self.w_q = np.random.randn(dim, dim).astype(np.float32) * 0.02
        self.w_k = np.random.randn(dim, dim).astype(np.float32) * 0.02
        self.w_v = np.random.randn(dim, dim).astype(np.float32) * 0.02
        self.w_o = np.random.randn(dim, dim).astype(np.float32) * 0.02

    def _softmax(self, x: np.ndarray):
        # TODO: 在最后一个维度做 softmax
        pass

    def forward(self, x: np.ndarray, past_kv=None):
        # TODO: 实现完整前向
        # 1. Q/K/V 投影: x @ w（注意形状转换）
        # 2. reshape 为 (batch, num_heads, seq, head_dim)
        # 3. 如果有 past_kv，沿 seq 维度拼接 K/V
        # 4. 计算 attention scores: Q @ K.T / sqrt(head_dim)
        # 5. Causal Mask（下三角）
        # 6. Softmax -> @ V -> reshape -> 输出投影
        # 返回: (output, (new_k, new_v))
        pass


class TestCausalSelfAttention:
    def test_without_pastkv(self):
        attn = CausalSelfAttention(dim=64, num_heads=4)
        x = np.random.randn(2, 5, 64).astype(np.float32)
        out, (k, v) = attn.forward(x, None)
        assert out.shape == (2, 5, 64)
        assert k.shape == (2, 4, 5, 16)

    def test_with_pastkv(self):
        attn = CausalSelfAttention(dim=64, num_heads=4)
        x1 = np.random.randn(2, 3, 64).astype(np.float32)
        out1, (k1, v1) = attn.forward(x1, None)

        x2 = np.random.randn(2, 1, 64).astype(np.float32)
        out2, (k2, v2) = attn.forward(x2, (k1, v1))

        assert out2.shape == (2, 1, 64)
        assert k2.shape == (2, 4, 4, 16)  # 3 + 1

    def test_empty_sequence(self):
        attn = CausalSelfAttention(dim=64, num_heads=4)
        x = np.random.randn(2, 0, 64).astype(np.float32)
        out, (k, v) = attn.forward(x, None)
        assert out.shape == (2, 0, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
