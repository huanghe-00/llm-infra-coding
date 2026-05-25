"""
【P06】BlockManager 综合题（带 fork + 引用计数）

要求：同 Q10，但 fork 后 parent 和 child 共享 block，
free 时只有引用计数归零才归还到 free_blocks。
"""

from typing import Dict, List, Set, Optional
import pytest


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        # TODO: 增加 ref_count 字典
        pass

    def allocate(self, req_id: str, seq_len: int) -> Optional[List[int]]:
        # TODO
        pass

    def free(self, req_id: str):
        # TODO: 引用计数减 1，归零才归还
        pass

    def fork(self, parent_req_id: str, child_req_id: str):
        # TODO: 引用计数加 1
        pass


class TestBlockManager:
    def test_ref_count(self):
        bm = BlockManager(num_blocks=5, block_size=16)
        bm.allocate("p", 20)  # 2 blocks
        bm.fork("p", "c1")
        bm.fork("p", "c2")

        # 此时引用计数应为 3，free c1 不应归还
        bm.free("c1")
        assert len(bm.free_blocks) == 3  # 5-2=3

        bm.free("c2")
        bm.free("p")
        assert len(bm.free_blocks) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
