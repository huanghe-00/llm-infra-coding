"""
【P03】动作 Tokenizer 综合题（7DoF + bins=256）

要求：同 Q08，但要求：
1. encode 返回的 token 在 [0, 255]
2. decode 后能还原到 bin 中心值
3. 提供 vocab_size 属性返回 256

C++ 程序员注意：
- @property 装饰器让方法像属性一样访问（不用括号）
"""

from typing import List
import pytest


class ActionTokenizer:
    def __init__(self, action_dim: int = 7, bins_per_dim: int = 256,
                 min_val: float = -1.0, max_val: float = 1.0):
        # TODO
        pass

    @property
    def vocab_size(self) -> int:
        # TODO
        pass

    def encode(self, action: List[float]) -> List[int]:
        # TODO
        pass

    def decode(self, tokens: List[int]) -> List[float]:
        # TODO
        pass


class TestActionTokenizer:
    def test_7dof(self):
        tok = ActionTokenizer(action_dim=7, bins_per_dim=256)
        action = [0.1] * 7
        tokens = tok.encode(action)
        assert len(tokens) == 7
        assert all(0 <= t < 256 for t in tokens)

    def test_vocab_size(self):
        tok = ActionTokenizer(bins_per_dim=256)
        assert tok.vocab_size == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
