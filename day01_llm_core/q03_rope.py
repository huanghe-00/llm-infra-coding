"""
【Q03】实现 RoPE（旋转位置编码）

要求：
1. 实现 RoPE 类，预计算 sin/cos 缓存
2. apply_rotary_emb(q, k, seq_len) 对 q/k 应用旋转编码
3. q/k 形状: (batch, num_heads, seq_len, head_dim)，head_dim 为偶数
4. 旋转矩阵：对每对 (d, d+1) 应用二维旋转，角度 m * theta_i，theta_i = 10000^(-2i/head_dim)

C++ 程序员注意：
- Python 的 complex number 可以用 np.view_as_complex，但建议用实数运算：
  q[..., 2i]   = q[..., 2i] * cos - q[..., 2i+1] * sin
  q[..., 2i+1] = q[..., 2i] * sin + q[..., 2i+1] * cos
- 注意 // 是整数除法，/ 是浮点除法
"""

import numpy as np
import pytest


class RoPE:
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        assert head_dim % 2 == 0
        # TODO: 预计算 inv_freq, sin_cache, cos_cache
        # sin_cache 形状: (max_seq_len, head_dim)
        pass

    def apply_rotary_emb(self, q: np.ndarray, k: np.ndarray, seq_len: int):
        # TODO: 截取前 seq_len 的 sin/cos，应用旋转，返回 (q_rot, k_rot)
        pass


class TestRoPE:
    def test_shape_preservation(self):
        rope = RoPE(head_dim=64)
        q = np.random.randn(2, 4, 10, 64).astype(np.float32)
        k = np.random.randn(2, 4, 10, 64).astype(np.float32)
        q_rot, k_rot = rope.apply_rotary_emb(q, k, 10)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotation_property(self):
        """相同 token 在不同位置，编码后点积应体现位置差异"""
        rope = RoPE(head_dim=64)
        q = np.ones((1, 1, 1, 64)).astype(np.float32)
        k = np.ones((1, 1, 1, 64)).astype(np.float32)

        # 位置 0
        q0, k0 = rope.apply_rotary_emb(q, k, 1)
        # 手动构造位置 1（seq_len=2，取第二个）
        q1 = np.ones((1, 1, 1, 64)).astype(np.float32)
        k1 = np.ones((1, 1, 1, 64)).astype(np.float32)
        _, full_k = rope.apply_rotary_emb(q1, np.concatenate([k1, k1], axis=2), 2)
        k1_rot = full_k[:, :, 1:2, :]

        # 位置 0 和位置 1 的编码结果应该不同
        assert not np.allclose(k0, k1_rot)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
