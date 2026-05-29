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
        # (batch) → (seq_len) → (num_heads) → (head_dim)
        # 公式 softmax_value = (e - max_x) / sum_e
        if x.shape[-1] == 0:
            return x
        x_max = np.max(x, axis = -1, keepdims = True)
        e = np.exp(x - x_max)
        x_sum = np.sum(e, axis = -1, keepdims = True)
        return e / x_sum 


    def forward(self, x: np.ndarray, past_kv=None): # self：指向类实例自身
        # TODO: 实现完整前向
        # 1. 计算QKV投影
        # x 形状: (batch, seq_len, dim)
        batchNum, seq_len, _ = x.shape

        # QKV形状 (batch, seq_len, dim) -> (batch, seq_len, num_heads, head_dim)
        Q = np.matmul(x, self.w_q)
        K = np.matmul(x, self.w_k)
        V = np.matmul(x, self.w_v)

        # 拆分多头 -> (batch, seq_len, num_heads, head_dim
        Q = np.reshape(Q, (batchNum, seq_len, self.num_heads, self.head_dim))
        K = np.reshape(K, (batchNum, seq_len, self.num_heads, self.head_dim))
        V = np.reshape(V, (batchNum, seq_len, self.num_heads, self.head_dim))

        # 调整seq_len, num_heads列位置 ->  (batch, num_heads, past_seq, head_dim)
        Q = np.transpose(Q, (0, 2, 1, 3))
        K = np.transpose(K, (0, 2, 1, 3))
        V = np.transpose(V, (0, 2, 1, 3))

        # 2. 取KV cache, 拼接
        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = K.shape[2]
            # 取出KV cache (batch, num_heads, past_seq, head_dim)
            K = np.concatenate([past_k, K], axis = 2)
            V = np.concatenate([past_v, V], axis = 2)
        new_K = K
        new_V = V

        # 3. 计算注意力 Q @ KT
        K_trans = np.transpose(K, (0, 1, 3, 2))
        scores = np.matmul(Q, K_trans) / np.sqrt(self.head_dim)

        # 4. mask矩阵掩码，通过某种方式得到一个掩码矩阵，使得每个token只能看到小于等于自己的注意力，在softmax前
        q_len = seq_len
        kv_len = K.shape[2]
        row_idx = np.arange(q_len)[:, None]
        col_idx = np.arange(kv_len)[None, :]
        mask = col_idx > (past_len + row_idx)
        scores = scores + mask * (-1e9)

        # 5. softmax
        scores_softmax = self._softmax(scores)

        # 6. @ V
        attention_weight = np.matmul(scores_softmax, V)

        # 7. 合并多头
        out = np.transpose(attention_weight, (0, 2, 1, 3))  #  (batch, num_heads, past_seq, head_dim) - >  (batch, past_seq, num_heads, head_dim)
        out = out.reshape(batchNum, seq_len, self.num_heads * self.head_dim)

        # 8. @ Wo
        out = np.matmul(out, self.w_o)
        return out, (new_K, new_V)






















        # 1. Q/K/V 投影: x @ w（注意形状转换）
        # 2. reshape 为 (batch, num_heads, seq, head_dim)
        # 3. 如果有 past_kv，沿 seq 维度拼接 K/V
        # 4. 计算 attention scores: Q @ K.T / sqrt(head_dim)
        # 5. Causal Mask（下三角）n
        # 6. Softmax -> @ V -> reshape -> 输出投影
        # 返回: (output, (new_k, new_v))
        # 把元组的前两个值赋给 batch 和 seq_len，第三个值赋给 _（_ 是 Python 惯例，表示“我不关心的值”）
        # 1. Q/K/V投影
        # 如果 a 或 b 是三维及以上的张量，会把最后两个维度当作矩阵，前面的维度当成 batch 维度，进行批量矩阵乘法。
        # 这里 x 形状 (batch, seq_len, dim)，w_q 形状 (dim, dim)，NumPy 自动把 x 的最后两维与 w_q 相乘，
        # 相当于对每个 batch 和每个 seq 位置独立做一次 (1, dim) @ (dim, dim) -> (1, dim)，最终形状 (batch, seq_len, dim)。


        # 2. 拆成多头并转置为 (batch, num_heads, seq, head_dim)


        # 交换数组的维度顺序
        # (batch, num_heads, seq_q, head_dim)
        

        # 3. 处理KV缓存


        # 4. 计算注意力分数
        # (batch, num_heads, head_dim, kv_len)
        # (batch, num_heads, q_len, kv_len)
        
        # 5. 因果掩码 (Causal Mask)
        # 目的：确保每个查询位置只能注意到“当前及之前”的键位置（含自身），不能看到未来。
        # 逻辑：
        #   - 当前查询索引 i ∈ [0, q_len)
        #   - 所有键索引   j ∈ [0, kv_len)    (kv_len = past_len + q_len)
        #   - 若 past_len > 0，则过去的键（索引 0..past_len-1）对任何 i 都可见
        #   - 对于新生成的键（索引 past_len .. kv_len-1），查询 i 只能看到 j <= past_len + i 的位置
        #   - 即：j > past_len + i 的位置必须屏蔽
        #
        # 下面的代码就是生成一个 (q_len, kv_len) 的布尔掩码矩阵，
        # 其中 True 表示“未来”位置（需要屏蔽），False 表示“历史或当前”位置（允许注意）。
        # 当前查询序列长度
        # 总的键序列长度 = past_len + q_len

        # np.arange(q_len) 生成 [0, 1, 2, ..., q_len-1]
        # 通过 [:, None] 在列方向增加一个维度，变成列向量，形状 (q_len, 1)
         # 代表每个查询的行索引

        # np.arange(kv_len) 生成 [0, 1, ..., kv_len-1]
        # 通过 [None, :] 在行方向增加一个维度，变成行向量，形状 (1, kv_len)
       # 代表每个键的列索引

        # 利用 NumPy 广播，比较生成 (q_len, kv_len) 的布尔矩阵
        # 条件：j > past_len + i
        #   - 当 j <= past_len + i 时，该位置为 False（可见）
        #   - 当 j > past_len + i 时，该位置为 True（未来，需屏蔽）
        

        # 将布尔 mask 转为浮点掩码：True 的位置乘以 -1e9，False 的位置乘以 0
        # -1e9 是一个极小的负数，在 softmax 中指数值约等于 0，从而实现屏蔽
        # 加到 scores 上，被屏蔽位置的分数变成极小值，其他位置不变
        
        # 6. Softmax（在最后一个维度 kv_len 上做）
        

        # 7. 加权求和
        

        # 8. 合并多头
        # 当前 out 形状: (batch, num_heads, q_len, head_dim)
        # 我们需要恢复成 (batch, q_len, dim)，即把多头合并回一个维度


        # 9. 输出投影




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
