"""
【P01】LRU Cache（LeetCode 146 风格）

要求：实现 LRUCache 类，支持 get 和 put，O(1) 时间复杂度。
用 OrderedDict 或手写双向链表 + HashMap。

C++ 程序员注意：
- OrderedDict.move_to_end(key) 把元素移到末尾（最新）
- OrderedDict.popitem(last=False) 弹出最老的
"""

from collections import OrderedDict
import pytest


class LRUCache:
    def __init__(self, capacity: int):
        # TODO
        pass

    def get(self, key: int) -> int:
        # TODO: 存在则返回值并移到最新，不存在返回 -1
        pass

    def put(self, key: int, value: int):
        # TODO: 插入/更新，超容量时淘汰最老的
        pass


class TestLRUCache:
    def test_basic(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        assert cache.get(1) == 1
        cache.put(3, 3)  # 淘汰 2
        assert cache.get(2) == -1
        cache.put(4, 4)  # 淘汰 1
        assert cache.get(1) == -1
        assert cache.get(3) == 3
        assert cache.get(4) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
