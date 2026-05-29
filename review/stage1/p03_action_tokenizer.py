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
        self.action_dim = action_dim
        self.bins_per_dim = bins_per_dim
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val
        self.bin_size = self.range / bins_per_dim

    @property
    def vocab_size(self) -> int:
        return self.bins_per_dim

    def encode(self, action: List[float]) -> List[int]:
        bin_list: List[int] = []
        for act in action:
            clipped = max(self.min_val, min(self.max_val, act))
            bin_idx = int((clipped - self.min_val) // self.bin_size)
            bin_idx = min(bin_idx, self.bins_per_dim - 1)  # 浮点数有误差，不能用==，此问题多次犯了，这里直接比较输出结果，越界则变为最大值
            bin_list.append(bin_idx)
        return bin_list

    def decode(self, tokens: List[int]) -> List[float]:
        action_list: List[float] = []
        for token in tokens:
            clipped_token = max(0, min(token, self.bins_per_dim))
            action = self.min_val + (clipped_token + 0.5) * self.bin_size
            action_list.append(action)
        return action_list


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
