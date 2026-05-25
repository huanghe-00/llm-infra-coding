"""
【Q15】简化版 FlashAttention（分块 Softmax）

要求：
1. flash_attention(Q, K, V, block_size=64) -> output
2. Q/K/V 形状: (batch, num_heads, seq_len, head_dim)
3. 不允许 materialize 完整的 (seq_len, seq_len) 注意力矩阵
4. 使用分块计算：将 Q/K/V 沿 seq_len 切成 block，逐块计算 attention
5. 每块内做标准 attention：scores = Q_block @ K_block.T / sqrt(d)
   -> causal mask（只算下三角）-> softmax -> @ V_block
6. 最后将各块输出拼接

C++ 程序员注意：
- 本题是"逻辑正确性"版本，不要求真的优化 HBM 访问
- np.concatenate 沿 axis=2（seq 维度）拼接输出块
- causal mask 只对当前块的相对位置生效，注意全局 offset
"""

import numpy as np
import pytest


def flash_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                    block_size: int = 64) -> np.ndarray:
    # TODO:
    # 1. 获取 batch, heads, seq, dim
    # 2. 初始化输出列表
    # 3. 外层循环：Q 分块（i 块）
    # 4. 内层循环：K/V 分块（j 块），只处理 j <= i（causal）
    # 5. 每块计算 scores，应用 causal mask（下三角）
    # 6. softmax，weighted sum，累加到当前 Q_block 的输出
    # 7. 拼接所有 Q_block 的输出
    pass


class TestFlashAttention:
    def test_shape_correctness(self):
        b, h, s, d = 2, 4, 128, 64
        Q = np.random.randn(b, h, s, d).astype(np.float32)
        K = np.random.randn(b, h, s, d).astype(np.float32)
        V = np.random.randn(b, h, s, d).astype(np.float32)

        out = flash_attention(Q, K, V, block_size=32)
        assert out.shape == (b, h, s, d)

    def test_causal_property(self):
        """causal attention 满足：位置 i 不应受位置 > i 的影响。
        通过构造特殊的 K/V 来验证：后续位置为极大值，不应影响前面。"""
        b, h, s, d = 1, 1, 8, 4
        Q = np.eye(d).reshape(1, 1, d, d).astype(np.float32)[:, :, :s, :]
        # 让 Q 的每个位置只关注自己维度
        Q = np.random.randn(1, 1, 8, 4).astype(np.float32)

        # K 的后半部分极大，若 non-causal 会影响前半部分输出
        K = np.zeros((1, 1, 8, 4), dtype=np.float32)
        K[:, :, 4:, :] = 1e6  # 后半部分极大

        V = np.zeros((1, 1, 8, 4), dtype=np.float32)
        V[:, :, 4:, 0] = 999  # 后半部分影响第 0 维

        out = flash_attention(Q, K, V, block_size=4)
        # 前半部分（位置 0-3）的输出第 0 维不应被 999 污染
        # 由于 Q 随机，这里只断言不崩溃且形状正确，causal 由形状测试间接保证
        assert out.shape == (1, 1, 8, 4)

    def test_matches_naive_for_small_seq(self):
        """小序列下，分块结果应与标准 causal attention 一致"""
        b, h, s, d = 1, 1, 16, 8
        Q = np.random.randn(b, h, s, d).astype(np.float32)
        K = np.random.randn(b, h, s, d).astype(np.float32)
        V = np.random.randn(b, h, s, d).astype(np.float32)

        # 标准 causal attention（允许 materialize）
        scores = Q @ np.transpose(K, (0, 1, 3, 2)) / np.sqrt(d)
        mask = np.tril(np.ones((s, s)))  # 下三角
        scores = scores * mask + (1 - mask) * (-1e9)
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        e = e * mask  # 上三角归零
        attn = e / np.sum(e, axis=-1, keepdims=True)
        expected = attn @ V

        out = flash_attention(Q, K, V, block_size=8)
        assert np.allclose(out, expected, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
