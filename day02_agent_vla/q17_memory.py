"""
【Q17】Agent 记忆系统（短期 buffer + 长期 top-k 检索）

要求：
1. Memory 类：
   - __init__(self, short_term_limit: int, embedding_dim: int)
   - add(message: str, embedding: np.ndarray) -> None
     同时存入短期 buffer（deque）和长期存储（list）
   - get_recent(k: int) -> List[str]：从短期 buffer 取最近 k 条
   - retrieve(query_embedding: np.ndarray, top_k: int) -> List[str]：
     从长期存储中计算余弦相似度，返回 top_k 条 message
2. 短期 buffer：固定长度，超限时丢弃最老的（deque maxlen）
3. 长期存储：不过期，全部保留
4. 余弦相似度：cos_sim(a,b) = (a·b) / (|a||b|)

C++ 程序员注意：
- collections.deque(maxlen=N) 自动丢弃超长的，不用手动 pop
- np.dot(a, b) 是点积
- np.linalg.norm(a) 是 L2 范数
- 存储结构建议：long_term = [{"msg": ..., "emb": ...}, ...]
"""

from typing import List
from collections import deque
import numpy as np
import pytest


class Memory:
    def __init__(self, short_term_limit: int, embedding_dim: int):
        # TODO:
        # 1. self.short_term = deque(maxlen=short_term_limit)
        # 2. self.long_term = []  # 元素为 dict 或 tuple
        # 3. 记录 embedding_dim 用于校验
        pass

    def add(self, message: str, embedding: np.ndarray):
        # TODO:
        # 1. 校验 embedding 形状
        # 2. 加入 short_term（自动 maxlen）
        # 3. 加入 long_term（append 即可，不过期）
        pass

    def get_recent(self, k: int) -> List[str]:
        # TODO: 从 short_term 取最后 k 条的 message
        # 注意：若 short_term 不足 k 条，返回全部
        pass

    def retrieve(self, query_embedding: np.ndarray, top_k: int) -> List[str]:
        # TODO:
        # 1. 若 long_term 为空，返回 []
        # 2. 计算 query_embedding 与每条长期记忆的余弦相似度
        # 3. 按相似度降序，取 top_k 条 message
        # 4. 若不足 top_k，返回全部
        pass


class TestMemory:
    def test_short_term_limit(self):
        mem = Memory(short_term_limit=3, embedding_dim=4)
        for i in range(5):
            mem.add(f"msg{i}", np.ones(4) * i)

        recent = mem.get_recent(10)
        assert recent == ["msg2", "msg3", "msg4"]

    def test_retrieve_topk(self):
        mem = Memory(short_term_limit=2, embedding_dim=2)
        mem.add("apple", np.array([1.0, 0.0]))
        mem.add("banana", np.array([0.9, 0.1]))
        mem.add("car", np.array([0.0, 1.0]))

        # 查询靠近 apple 的向量
        results = mem.retrieve(np.array([1.0, 0.0]), top_k=2)
        assert "apple" in results
        assert len(results) == 2

    def test_retrieve_empty(self):
        mem = Memory(short_term_limit=2, embedding_dim=2)
        results = mem.retrieve(np.array([1.0, 0.0]), top_k=2)
        assert results == []

    def test_recent_less_than_k(self):
        mem = Memory(short_term_limit=5, embedding_dim=2)
        mem.add("a", np.zeros(2))
        recent = mem.get_recent(10)
        assert recent == ["a"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
