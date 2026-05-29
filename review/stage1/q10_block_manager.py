"""
【Q10】简化版 PagedAttention BlockManager

要求：
1. BlockAllocator(num_blocks, block_size)
2. allocate(req_id, seq_len) -> List[int] 或 None（分配 block_ids）
3. free(req_id) -> 释放该请求占用的所有 block
4. fork(parent_req_id, child_req_id) -> 复制 block_table（写时复制，这里直接复制引用）
5. 计算需要 block 数: ceil(seq_len / block_size)

C++ 程序员注意：
- set.pop() 任意取出一个元素
- dict.get(key) 安全访问
- 复制 list: list(old_list) 或 old_list.copy()（浅拷贝，本题够用）
"""

from typing import Dict, List, Set, Optional
import pytest


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        # TODO: 初始化 free_blocks(set), block_tables(dict)
        # vllm预分配一块内存，长度为num_blocks * block_size
        # 当前类是管理内存中虚拟ID(偏移量)的管理方式
        # 使用set类型——去重无序的容器，管理闲置/已释放ID
        # 使用dict类型——作为块表，管理req_id和分配块id list的关系
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: set[int] = set(range(self.num_blocks))
        self.block_table: dict[str, list[int]] = {}

    def allocate(self, req_id: str, seq_len: int) -> Optional[List[int]]:
        # TODO: 计算 need_blocks，从 free_blocks 分配，记录到 block_tables
        need_blocks = (seq_len + self.block_size - 1) // self.block_size
        if need_blocks > len(self.free_blocks):
            return None

        allocated_id: list[int] = [self.free_blocks.pop() for _ in range(need_blocks)]
        self.block_table[req_id] = allocated_id
        return allocated_id

    def free(self, req_id: str):
        # TODO: 释放 block，归还到 free_blocks
        if req_id not in self.block_table:
            return
        free_list: list[int] = self.block_table.get(req_id, [])
        for block_id in free_list:
            self.free_blocks.add(block_id)
        self.block_table.pop(req_id)
        return

    def fork(self, parent_req_id: str, child_req_id: str) -> Optional[List[int]]:
        # TODO: 复制 parent 的 block_table 给 child
        if parent_req_id not in self.block_table:
            return None
        if child_req_id in self.block_table:
            self.free(child_req_id)
        parent_blocks: list[int] = self.block_table.get(parent_req_id, [])
        self.block_table[child_req_id] = list(parent_blocks)
        return self.block_table[child_req_id]


class TestBlockAllocator:
    def test_basic_allocate(self):
        alloc = BlockAllocator(num_blocks=10, block_size=16)
        blocks = alloc.allocate("req_1", 20)  # need 2 blocks
        assert blocks is not None and len(blocks) == 2

    def test_free_and_reuse(self):
        alloc = BlockAllocator(num_blocks=3, block_size=16)
        b1 = alloc.allocate("req_1", 20)  # 2 blocks
        alloc.free("req_1")
        b2 = alloc.allocate("req_2", 20)
        assert b2 is not None and len(b2) == 2

    def test_oom(self):
        alloc = BlockAllocator(num_blocks=2, block_size=16)
        alloc.allocate("req_1", 40)  # need 3, 失败
        b = alloc.allocate("req_2", 40)
        assert b is None

    def test_fork(self):
        alloc = BlockAllocator(num_blocks=10, block_size=16)
        b1 = alloc.allocate("parent", 30)  # 2 blocks
        b2 = alloc.fork("parent", "child")
        assert b2 is not None and len(b2) == 2
        assert b2 == b1  # 本题直接引用复制


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
