"""
【Q19】带优先级的请求调度器（机器人急停优先）

要求：
1. PriorityScheduler(max_batch_size: int)
2. add_request(req_id: str, priority: int, prompt_len: int)
   - priority: 0=急停(EMERGENCY), 1=动作(ACTION), 2=查询(QUERY)
3. step() -> List[str]：
   - 返回当前 batch 的 req_id 列表
   - 急停请求无条件插队：即使 batch 已满，也要踢出一个最低优先级的请求（QUERY -> ACTION），把急停加进来
   - 同优先级按 FIFO
   - 每次 step() 后，batch 中所有请求的 remaining_tokens -= 1
   - remaining_tokens 初始 = prompt_len，到 0 时请求完成，从 batch 移除
4. is_done(req_id) -> bool：该请求是否已完成

C++ 程序员注意：
- 等待队列用 list 即可，每次 step() 前排序（priority 小的在前）
- batch 用 dict 或 list 维护，记录 {req_id: remaining}
- 踢出时找 batch 中 priority 最大的（数值最大=优先级最低）且非急停的请求
- Python list 的 pop(i) 是 O(n)，但本题 batch_size 很小，够用
"""

from typing import List, Dict
import pytest


class PriorityScheduler:
    def __init__(self, max_batch_size: int):
        # TODO:
        # 1. self.max_batch_size = max_batch_size
        # 2. self.waiting = []  # 元素 (req_id, priority, remaining)
        # 3. self.batch = {}  # req_id -> remaining，或更复杂的结构
        # 4. 需要同时记录 req_id -> priority，用于踢出判断
        pass

    def add_request(self, req_id: str, priority: int, prompt_len: int):
        # TODO: 加入 waiting 队列
        pass

    def step(self) -> List[str]:
        # TODO:
        # 1. batch 中所有 remaining -= 1，移除到 0 的
        # 2. 从 waiting 按优先级补充 batch，直到满
        # 3. 若 waiting 中有 priority=0（急停）且 batch 已满：
        #    找到 batch 中 priority 最大（最低）的请求踢出，放回 waiting 头部
        #    把急停加入 batch
        # 4. 返回当前 batch 的 req_id 列表
        pass

    def is_done(self, req_id: str) -> bool:
        # TODO: 请求不在 batch 且不在 waiting 且曾经加入过 -> 已完成
        # 简化：维护一个 completed 集合
        pass


class TestPriorityScheduler:
    def test_basic_priority(self):
        s = PriorityScheduler(max_batch_size=2)
        s.add_request("q1", priority=2, prompt_len=3)  # QUERY
        s.add_request("a1", priority=1, prompt_len=3)  # ACTION
        s.add_request("e1", priority=0, prompt_len=2)  # EMERGENCY

        batch1 = s.step()
        assert "e1" in batch1  # 急停必须进入
        assert "q1" not in batch1  # QUERY 被踢出或没进

    def test_emergency_preempt(self):
        s = PriorityScheduler(max_batch_size=1)
        s.add_request("a1", priority=1, prompt_len=5)
        s.step()  # a1 进入 batch

        s.add_request("e1", priority=0, prompt_len=2)
        batch = s.step()
        assert "e1" in batch
        assert "a1" not in batch  # 被踢出

    def test_done(self):
        s = PriorityScheduler(max_batch_size=2)
        s.add_request("a1", priority=1, prompt_len=1)
        s.step()  # a1 进入，remaining=1
        s.step()  # remaining=0，应完成
        assert s.is_done("a1") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
