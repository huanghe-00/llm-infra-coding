"""
【Q08】实现机械臂动作 Tokenization（连续 -> 离散 bin）

要求：
1. ActionTokenizer(action_dim, bins_per_dim, min_val=-1.0, max_val=1.0)
2. encode(action: List[float]) -> List[int]：clip -> 均匀分 bin -> int
3. decode(tokens: List[int]) -> List[float]：取 bin 中心值
4. bin 数量 = bins_per_dim，取 2 的幂更友好（如 256）
5. 越界值 clip 到 [min_val, max_val]

C++ 程序员注意：
- Python 的 int() 是向零截断，对于正数等价于 floor
- 计算 bin index: int((clipped - min) / bin_size)
- 边界保护：min(idx, bins-1)
"""

from typing import List
import pytest


class ActionTokenizer:
    def __init__(self, action_dim: int, bins_per_dim: int,
                 min_val: float = -1.0, max_val: float = 1.0):
        # TODO: 初始化参数，计算 bin_size
        self._action_dim = action_dim          # 动作维度，类似 C++ private 成员
        self._bins = bins_per_dim              # 每维分桶数量，推荐 2 的幂
        self._min_val = min_val                # 连续值下界
        self._max_val = max_val                # 连续值上界

        BIN_COUNT = self._bins                 # 局部常量，避免 magic number
        RANGE = self._max_val - self._min_val  # 连续值范围
        self._bin_size = RANGE / BIN_COUNT     # 单 bin 宽度，单位：连续值单位/bin

    def encode(self, action: List[float]) -> List[int]:
        # TODO: clip -> bin -> int
        """将连续动作序列 clip 并离散化为 token 序列"""
        tokens: List[int] = []
        # 边界保护常量
        BIN_MAX_INDEX = self._bins - 1

        for value in action:
            # 1. Clip到有效区间  设计原因 硬实时系统要“吞掉”野值，而不是“炸掉”流程。
            clipped = max(self._min_val, min(value, self._max_val))
        
            # 2. 计算Bin索引--量化
            index = int((clipped - self._min_val) / self._bin_size)

            # 3. 边界保护
            safe_index = min(index, BIN_MAX_INDEX)

            tokens.append(safe_index)
        
        return tokens

    def decode(self, tokens: List[int]) -> List[float]:
        # TODO: int -> bin 中心值
        """将 token 序列还原为连续值（取 bin 中心）"""
        actions: List[float] = []

        CENTER_OFFSET = 0.5

        for token in tokens:
            center = self._min_val + (token + CENTER_OFFSET) * self._bin_size
            actions.append(center)

        return actions

class TestActionTokenizer:
    def test_encode_decode_roundtrip(self):
        tok = ActionTokenizer(action_dim=7, bins_per_dim=256)
        action = [0.5, -0.3, 0.0, 0.99, -0.99, 0.1, -0.1]
        tokens = tok.encode(action)
        assert len(tokens) == 7
        assert all(0 <= t < 256 for t in tokens)

        decoded = tok.decode(tokens)
        for a, d in zip(action, decoded):
            assert abs(a - d) < 0.02  # 量化误差容忍

    def test_clip_boundary(self):
        tok = ActionTokenizer(action_dim=2, bins_per_dim=10,
                              min_val=-1.0, max_val=1.0)
        tokens = tok.encode([1.5, -2.0])
        assert tokens[0] == 9   # max bin
        assert tokens[1] == 0   # min bin

    def test_uniform_bins(self):
        tok = ActionTokenizer(action_dim=1, bins_per_dim=4,
                              min_val=0.0, max_val=4.0)
        # bin 大小 = 1.0，中心: 0.5, 1.5, 2.5, 3.5
        tokens = tok.encode([0.5, 1.5, 2.5, 3.5])
        assert tokens == [0, 1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
