"""
【P05】Top-K + Top-P 采样综合题

要求：同 Q04，但增加：
1. 如果 top_k=0 且 top_p=1.0，等价于纯 temperature 采样
2. 如果 temperature=0，无视 top_k/top_p，直接 greedy
3. 输入 logits 可以是 2D (batch, vocab)，返回 List[int]

C++ 程序员注意：
- 对 2D 数组，沿 axis=-1（最后一个维度）操作
"""

import numpy as np
import pytest


class Sampler:
    def __init__(self, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
        # TODO
        pass

    def sample(self, logits: np.ndarray) -> List[int]:
        # TODO: 支持 batch 维度
        pass


class TestSampler:
    def test_batch_greedy(self):
        sampler = Sampler(temperature=0.0)
        logits = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        result = sampler.sample(logits)
        assert result == [2, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
