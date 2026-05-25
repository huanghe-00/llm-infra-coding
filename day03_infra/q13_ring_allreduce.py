"""
【Q13】Ring AllReduce 模拟（4 节点）

要求：
1. ring_allreduce(gradients_list: List[np.ndarray]) -> List[np.ndarray]
2. gradients_list[i] 是第 i 个节点的梯度（形状相同）
3. 模拟 Ring AllReduce 两阶段：
   - Scatter-Reduce：每个节点沿环发送 chunk，接收后累加
   - All-Gather：每个节点沿环发送完整结果，接收后拼接
4. 简化：直接计算全局平均，返回每个节点一份完整平均梯度
   （真实实现是分 chunk 通信，笔试通常考"逻辑正确性"而非真实通信模拟）

C++ 程序员注意：
- sum(list) / len(list) 就是平均
- 返回 [avg_grad] * n 会让所有节点共享同一个引用，应该返回 [avg_grad.copy() for _ in range(n)]
"""

from typing import List
import numpy as np
import pytest


def ring_allreduce(gradients_list: List[np.ndarray]) -> List[np.ndarray]:
    # TODO: 计算全局平均梯度，每个节点返回一份 copy
    pass


class TestRingAllReduce:
    def test_4_nodes(self):
        g0 = np.array([1.0, 2.0, 3.0])
        g1 = np.array([2.0, 3.0, 4.0])
        g2 = np.array([3.0, 4.0, 5.0])
        g3 = np.array([4.0, 5.0, 6.0])

        result = ring_allreduce([g0, g1, g2, g3])
        expected = np.array([2.5, 3.5, 4.5])  # 平均

        assert len(result) == 4
        for r in result:
            assert np.allclose(r, expected)

    def test_independent_copy(self):
        g = [np.ones(3), np.ones(3) * 2]
        result = ring_allreduce(g)
        result[0][0] = 999
        # 修改 result[0] 不应影响 result[1]
        assert result[1][0] != 999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
