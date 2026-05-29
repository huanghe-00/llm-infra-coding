"""
【P09】Attention + past_kv 综合题（支持 GQA）

要求：结合 Q02 和 Q05，实现支持 past_kv 的 GQA Attention。
Q 有 num_heads，K/V 有 num_kv_heads，通过广播计算 attention。
"""

import numpy as np
import pytest


class GQAAttention:
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        assert num_heads % num_kv_heads == 0
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_group = num_heads // num_kv_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        # 模拟参数矩阵
        scale = 0.02
        self.w_q = np.random.randn(dim, num_heads * self.head_dim).astype(np.float32) * scale
        self.w_k = np.random.randn(dim, num_kv_heads * self.head_dim).astype(np.float32) * scale
        self.w_v = np.random.randn(dim, num_kv_heads * self.head_dim).astype(np.float32) * scale
        self.w_o = np.random.randn(num_heads * self.head_dim, dim).astype(np.float32) * scale

    
    def forward(self, x: np.ndarray, past_kv = None):
        # TODO
        batchNum, seq_len, x_dim = x.shape
        # 1. 计算Q K V
        Q = np.matmul(x, self.w_q)
        K = np.matmul(x, self.w_k)
        V = np.matmul(x, self.w_v)
        # 拆分多头
        Q = Q.reshape(batchNum, seq_len, self.num_heads, self.head_dim)
        Q = np.transpose(Q, (0, 2, 1, 3))
        K = K.reshape(batchNum, seq_len, self.num_kv_heads, self.head_dim)
        K = np.transpose(K, (0, 2, 1, 3))
        V = V.reshape(batchNum, seq_len, self.num_kv_heads, self.head_dim)
        V = np.transpose(V, (0, 2, 1, 3))  # (batchNum, self.num_kv_heads, seq_len, self.head_dim)
        # 2. 取kv_cache
        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[2]
            K = np.concatenate([past_k, K], axis = 2)
            V = np.concatenate([past_v, V], axis = 2)
        new_kv = (K.copy(), V.copy())  # 这里拷贝的原因是，之前MHA MGA不会再修改KV，而这里接下来就做了repeat操作

        # 3. 分组广播，复制多头共享的K V头  # (batchNum, self.num_kv_heads, seq_len, self.head_dim)
        # GQA核心差别，MHA和MQA都没有这一步，由于每self.num_group共用一个kv，需要在axis=2维度上，复制group个kv
        K = np.repeat(K, repeats = self.num_group, axis = 1) 
        V = np.repeat(V, repeats = self.num_group, axis = 1)
    
        # 4. attention = Q @ KT / sqrt(head_dim)
        K_trans = np.transpose(K, (0, 1, 3, 2))
        attention = np.matmul(Q, K_trans) / np.sqrt(self.head_dim)

        # 5. mask掩码
        # 建立一个seq_len行 past_kv + kv_len 列的矩阵
        # 历史序号 history_id = row_id + past_kv
        # 目标是放行所有 history_id <= col_id的矩阵，因此mask条件是 col_id > row_id + past_len
        row_id = np.arange(seq_len)[:, None]
        kv_len = K.shape[2]
        col_id = np.arange(kv_len)[None, :]
        mask = col_id > row_id + past_len
        attention += mask * (-1e9)

        # 6. softmax
        attention_max = np.max(attention, axis = -1, keepdims = True)
        attention_e = np.exp(attention - attention_max)
        attention_softmax = attention_e / np.sum(attention_e, axis = -1, keepdims = True)

        # 7. scores = attention @ V
        scores = np.matmul(attention_softmax, V)

        # 8. 合并多头  要将所有head拼接成num_head * head_dim
        attention_weight = np.transpose(scores, (0, 2, 1, 3))
        attention_weight = attention_weight.reshape(batchNum, seq_len, self.head_dim * self.num_heads)

        # 9. out = out @ W_o
        out = np.matmul(attention_weight, self.w_o)
        return out, new_kv

class TestGQAAttention:
    def test_gqa_pastkv(self):
        attn = GQAAttention(dim=64, num_heads=8, num_kv_heads=2)
        x1 = np.random.randn(1, 2, 64).astype(np.float32)
        out1, (k1, v1) = attn.forward(x1, None)

        x2 = np.random.randn(1, 1, 64).astype(np.float32)
        out2, (k2, v2) = attn.forward(x2, (k1, v1))

        assert k2.shape == (1, 2, 3, 8)  # head_dim=8
    def test_gqa_pastkv_2(self):
        np.random.seed(42)
        attn = GQAAttention(dim=64, num_heads=8, num_kv_heads=2)
        x1 = np.random.randn(1, 2, 64).astype(np.float32)
        out1, (k1, v1) = attn.forward(x1, None)

        x2 = np.random.randn(1, 1, 64).astype(np.float32)
        out2, (k2, v2) = attn.forward(x2, (k1, v1))

        # head_dim = 64/8 = 8, total seq_len = 2+1=3, kv heads=2
        assert k2.shape == (1, 2, 3, 8), f"Expected (1,2,3,8) but got {k2.shape}"
        assert out2.shape == (1, 1, 64)

    def test_forward_shape(self):
        """单次前向，无 past_kv"""
        np.random.seed(0)
        attn = GQAAttention(dim=32, num_heads=4, num_kv_heads=2)
        x = np.random.randn(2, 5, 32).astype(np.float32)
        out, (k, v) = attn.forward(x, None)
        assert out.shape == (2, 5, 32)
        # kv 缓存形状应为 (batch, num_kv_heads, seq_len, head_dim)
        assert k.shape == (2, 2, 5, 8)
        assert v.shape == (2, 2, 5, 8)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
