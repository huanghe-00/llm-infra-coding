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
        # 数组的外层 → 内层
        # (batch) → (num_heads) → (q_len) → (head_dim)
        max_x = np.max(x, axis = -1, keepdims = True)
        e_x = np.exp(x - max_x)
        sum_e = np.sum(e_x, axis = -1, keepdims = True)
        return e_x / sum_e

    def forward(self, x: np.ndarray, past_kv=None): # self：指向类实例自身
        # TODO: 实现完整前向
        # 1. Q/K/V 投影: x @ w（注意形状转换）
        # 2. reshape 为 (batch, num_heads, seq, head_dim)
        # 3. 如果有 past_kv，沿 seq 维度拼接 K/V
        # 4. 计算 attention scores: Q @ K.T / sqrt(head_dim)
        # 5. Causal Mask（下三角）n
        # 6. Softmax -> @ V -> reshape -> 输出投影
        # 返回: (output, (new_k, new_v))

        batch, seq_len, _ = x.shape # 把元组的前两个值赋给 batch 和 seq_len，第三个值赋给 _（_ 是 Python 惯例，表示“我不关心的值”）
        # 1. Q/K/V投影
        # 如果 a 或 b 是三维及以上的张量，会把最后两个维度当作矩阵，前面的维度当成 batch 维度，进行批量矩阵乘法。
        # 这里 x 形状 (batch, seq_len, dim)，w_q 形状 (dim, dim)，NumPy 自动把 x 的最后两维与 w_q 相乘，
        # 相当于对每个 batch 和每个 seq 位置独立做一次 (1, dim) @ (dim, dim) -> (1, dim)，最终形状 (batch, seq_len, dim)。
        Q = np.matmul(x, self.w_q)
        K = np.matmul(x, self.w_k)
        V = np.matmul(x, self.w_v)

        # 2. 拆成多头并转置为 (batch, num_heads, seq, head_dim)
        Q.reshape(batch, seq_len, self.nums_heads, self.head_dim)
        K.reshape(batch, seq_len, self.nums_heads, self.head_dim)
        V.reshape(batch, seq_len, self.nums_heads, self.head_dim)

        Q.transpose(Q, (0, 2, 1, 3)) # 交换数组的维度顺序
        K.transpose(K, (0, 2, 1, 3)) # (batch, num_heads, seq_q, head_dim)
        V.transpose(V, (0, 2, 1, 3))

        # 3. 处理KV缓存
        if past_kv is not None:
            past_K, past_V = past_kv
            # 沿着seq维度拼接
            K = np.concatencate([past_K, K], axis = 2)
            V = np.concatencate([past_V, V], axis = 2)
            past_len = K.shape[2]
        new_k = K
        new_v = V

        # 4. 计算注意力分数
        K_trans = np.transpose(K, (0, 1, 3, 2)) # (batch, num_heads, head_dim, kv_len)
        scores = np.matmul(Q, K_trans) / np.sqrt(self.head_dim) # (batch, num_heads, q_len, kv_len)
        
        # 5. 因果掩码
        q_len = seq_len
        kv_len = K.shape[2]
        


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
