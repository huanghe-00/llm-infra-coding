"""
【Q09】实现简化版端到端 VLA 前向流程

要求：
1. SimpleVLA 类，process(image_tokens, text_tokens) -> action_logits
2. 模拟结构：
   - image_proj: Linear(image_dim -> hidden_dim)
   - text_embed: Linear(vocab_size -> hidden_dim) 或直接用输入作为 embedding
   - fusion: 拼接 image_tokens 和 text_tokens，过 Transformer（简化：只过一层 Attention）
   - action_head: Linear(hidden_dim -> action_vocab_size)
3. 输入：
   - image_tokens: (batch, num_image_tokens, image_dim)
   - text_tokens: (batch, seq_len, hidden_dim)  已 embedding 化
4. 输出: (batch, action_vocab_size)

C++ 程序员注意：
- 本题是"模拟" VLA 流程，不要求真实 Transformer 参数
- 用随机矩阵模拟投影，输出形状正确即可
- np.concatenate 沿 seq 维度拼接（axis=1）
"""

import numpy as np
import pytest


class SimpleVLA:
    def __init__(self, image_dim: int, hidden_dim: int, action_vocab_size: int,
                 num_heads: int = 4):
        self.hidden_dim = hidden_dim
        self.action_vocab_size = action_vocab_size

        np.random.seed(42)
        # TODO: 初始化 image_proj, action_head
        # 可选：一层简化 Attention 参数
        pass

    def process(self, image_tokens: np.ndarray, text_tokens: np.ndarray) -> np.ndarray:
        # TODO:
        # 1. image_tokens (b, n_img, image_dim) -> project -> (b, n_img, hidden_dim)
        # 2. 与 text_tokens (b, seq, hidden_dim) 沿 seq 拼接 -> (b, n_img+seq, hidden_dim)
        # 3. 过一层简化 Attention（或直接取 mean/max pooling）
        # 4. action_head -> (b, action_vocab_size)
        pass


class TestSimpleVLA:
    def test_output_shape(self):
        vla = SimpleVLA(image_dim=512, hidden_dim=256, action_vocab_size=100)
        img = np.random.randn(2, 4, 512).astype(np.float32)
        txt = np.random.randn(2, 10, 256).astype(np.float32)
        logits = vla.process(img, txt)
        assert logits.shape == (2, 100)

    def test_single_batch(self):
        vla = SimpleVLA(image_dim=128, hidden_dim=64, action_vocab_size=16)
        img = np.random.randn(1, 2, 128).astype(np.float32)
        txt = np.random.randn(1, 5, 64).astype(np.float32)
        logits = vla.process(img, txt)
        assert logits.shape == (1, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
