"""
【Q04】实现文本生成采样策略（Greedy / Top-K / Top-P / Temperature）

要求：
1. 实现 sample(logits, temperature, top_k, top_p) -> int (token_id)
2. logits: 1D numpy 数组，长度 vocab_size
3. temperature: 除法缩放（0 表示 greedy）
4. top_k: 保留概率最高的 k 个，其余置 -inf
5. top_p (nucleus): 按概率降序累加，保留最小集合使累加概率 >= top_p
6. 如果 temperature == 0，直接返回 argmax（greedy）

C++ 程序员注意：
- np.argsort 返回排序后的索引，相当于 C++ 的 sort + 索引跟踪
- np.cumsum 是前缀和
- 随机采样用 np.random.choice，传入概率分布 p
"""

import numpy as np
import pytest


def sample(logits: np.ndarray, temperature: float = 1.0,
           top_k: int = 0, top_p: float = 1.0) -> int:
    # TODO: 实现采样逻辑
    # 1. temperature 缩放（注意 temperature=0 时直接 greedy）
    # 2. top_k 过滤
    # 3. top_p (nucleus) 过滤
    # 4. softmax
    # 5. 采样返回 token_id
    pass


class TestSampling:
    def test_greedy(self):
        logits = np.array([1.0, 2.0, 3.0, 0.5])
        token = sample(logits, temperature=0.0)
        assert token == 2  # argmax

    def test_temperature_effect(self):
        """高温使分布更均匀，低温更尖锐。这里只测不崩溃"""
        logits = np.array([1.0, 2.0, 3.0])
        for _ in range(10):
            t = sample(logits, temperature=0.5)
            assert 0 <= t < 3

    def test_top_k(self):
        logits = np.array([10.0, 1.0, 1.0, 1.0])
        # 多次采样，top_k=1 应该总是选到第一个
        for _ in range(20):
            t = sample(logits, temperature=1.0, top_k=1)
            assert t == 0

    def test_top_p(self):
        logits = np.array([5.0, 4.0, 0.1, 0.1])
        # top_p=0.5 应该只保留前两个（概率占比极高）
        results = [sample(logits, temperature=1.0, top_p=0.5) for _ in range(50)]
        assert all(r in [0, 1] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
