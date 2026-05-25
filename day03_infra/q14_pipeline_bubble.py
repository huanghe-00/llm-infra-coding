"""
【Q14】流水线并行气泡时间计算

要求：
1. calculate_bubble(stages: int, micro_batches: int, p2p_time: float = 1.0) -> (bubble_time, bubble_rate)
2. 公式：
   - 总时间 = (stages + micro_batches - 1) * p2p_time  （简化版，假设每阶段时间相等）
   - 理想时间（无气泡）= micro_batches * p2p_time
   - bubble_time = 总时间 - 理想时间 = (stages - 1) * p2p_time
   - bubble_rate = bubble_time / 总时间
3. 如果 stages=1，bubble_time=0，bubble_rate=0

C++ 程序员注意：
- 注意 float 除法，Python 3 的 / 就是浮点除
- 返回 tuple: (bubble_time, bubble_rate)
"""

import pytest


def calculate_bubble(stages: int, micro_batches: int, p2p_time: float = 1.0):
    # TODO: 按公式计算
    pass


class TestPipelineBubble:
    def test_single_stage(self):
        bt, br = calculate_bubble(stages=1, micro_batches=4)
        assert bt == 0.0
        assert br == 0.0

    def test_basic(self):
        # stages=4, micro_batches=8
        # total = (4 + 8 - 1) * 1 = 11
        # ideal = 8
        # bubble = 3, rate = 3/11
        bt, br = calculate_bubble(stages=4, micro_batches=8)
        assert bt == 3.0
        assert abs(br - 3.0 / 11.0) < 1e-6

    def test_large_micro_batches(self):
        # micro_batches >> stages 时，bubble_rate 应该很小
        bt, br = calculate_bubble(stages=4, micro_batches=100)
        assert br < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
