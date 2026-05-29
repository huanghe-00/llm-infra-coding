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
        x_max = np.max(x, axis = -1, keepdims = True)  # argmax返回索引，max返回值
        e = np.exp(x - x_max)
        return e / np.sum(e, axis = -1, keepdims = True)

    def forward(self, x: np.ndarray, past_kv=None):
        # TODO: 实现 MQA 前向
        # 1. Q 投影 reshape -> (batch, num_heads, seq, head_dim)
        # 2. K/V 投影 reshape -> (batch, 1, seq, head_dim)
        # 3. 拼接 past_kv
        # 4. Attention: Q @ K.T，K 广播到 num_heads
        # 5. Causal Mask, Softmax, @ V，V 广播
        # 6. 输出投影

        batchNum, seq_len, x_dim = x.shape
        print(f"{batchNum}, {seq_len}, {x_dim}")
        # 0. 计算投影Q K V，MHA的特例，num_head = 1
        # 初始 x 的状态 (batch, seq, head_dim)  Q (batch, num_heads, seq, head_dim) KV  (batch, 1, seq, head_dim)
        Q = np.matmul(x, self.w_q)
        K = np.matmul(x, self.w_k)
        V = np.matmul(x, self.w_v)

        # reshape->transpose
        Q = Q.reshape(batchNum, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batchNum, seq_len, 1, self.head_dim)
        V = V.reshape(batchNum, seq_len, 1, self.head_dim)
        Q = np.transpose(Q, (0, 2, 1, 3))
        K = np.transpose(K, (0, 2, 1, 3))
        V = np.transpose(V, (0, 2, 1, 3))

        # 1. kv_cache
        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_v.shape[2]
            K = np.concatenate([past_k, K], axis = 2)
            V = np.concatenate([past_v, V], axis = 2)
        new_K = K
        new_V = V

        # 2. 计算Q @ KT   (batch, seq, head_dim, 1)
        K_trans = np.transpose(K, (0, 1, 3, 2)) 
        scores = np.matmul(Q, K_trans) / np.sqrt(self.head_dim)

        # 3. 计算mask\
        q_len = Q.shape[2]
        kv_len = K.shape[2]
        row_idx = np.arange(q_len)[:, None]  # 形状(q_len, 1)     假设q_len = 3  [[0], [1], [2]]
        col_idx = np.arange(kv_len)[None, :] # 形状(1, kv_len)    假设kv_len = 5 [[0, 1, 2, 3, 4]]
        print(f"past_len:{past_len}, kv_len:{kv_len}, q_len:{q_len}")
        # 理解方式，本轮的row_id = 0 -> 历史序列row_id = 0 + past_len = 2
        # 那么它能看到0,1,2   看不到3，4
        # 因此 row_id + past_len <= col_id 可见 row_id + past_len > col_id 不可见
        mask = col_idx > row_idx + past_len
        scores += mask * (-1e9)

        # 4. 计算softmax
        softmax_scores = self._softmax(scores)

        # 5. 计算attenttion_weight = softmax @ V (batch, seq, head_dim, 1)
        attention_weight = np.matmul(softmax_scores, V)

        # 7. 合并多头，输出（如果有多头）
        attention_weight = np.transpose(attention_weight, (0, 2, 1, 3))
        attention_weight = attention_weight.reshape(batchNum, seq_len, x_dim)

        # 6. 计算out = attention_weight @ W_o
        out = np.matmul(attention_weight, self.w_o)
        return out, (new_K, new_V)


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
