"""
【Q11】请求调度器（Continuous Batching 模拟）

要求：
1. Scheduler(max_batch_size)
2. add_request(req_id, prompt_len) -> 加入等待队列
3. step() -> 返回当前 batch 中的 req_id 列表
4. 调度策略：
   - 优先处理等待队列中的 prefill 请求（新请求）
   - 如果当前 batch 未满，加入等待请求
   - 如果当前 batch 中某请求已完成（假设模拟完成标记），移除
5. 简化：每个请求有一个 remaining_tokens 计数，step() 时减 1，到 0 时完成

C++ 程序员注意：
- 用 collections.deque 做等待队列（popleft 是 O(1)）
- 当前 batch 用 dict 或 list 维护
"""

from typing import List, Dict
from collections import deque
import pytest


class Scheduler:
    def __init__(self, max_batch_size: int):
        # TODO: 初始化等待队列、当前 batch、请求状态
        pass

    def add_request(self, req_id: str, prompt_len: int):
        # TODO: 加入等待队列，记录剩余 token 数
        pass

    def step(self) -> List[str]:
        # TODO:
        # 1. 检查当前 batch 中哪些请求已完成，移除
        # 2. 从等待队列补充新请求，直到 batch 满
        # 3. 对所有当前 batch 中的请求，remaining_tokens -= 1
        # 4. 返回当前 batch 的 req_id 列表
        pass


class TestScheduler:
    def test_basic_batching(self):
        s = Scheduler(max_batch_size=2)
        s.add_request("a", 3)
        s.add_request("b", 3)
        s.add_request("c", 3)

        batch1 = s.step()
        assert set(batch1) == {"a", "b"}

    def test_continuous_add(self):
        s = Scheduler(max_batch_size=2)
        s.add_request("a", 1)
        batch = s.step()
        assert "a" in batch

        # a 已完成，此时加入 b，下一步应该只有 b
        s.add_request("b", 2)
        batch2 = s.step()
        assert "a" not in batch2
        assert "b" in batch2

    def test_batch_size_limit(self):
        s = Scheduler(max_batch_size=1)
        s.add_request("a", 5)
        s.add_request("b", 5)
        batch = s.step()
        assert len(batch) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
