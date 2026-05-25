"""
【Q20】VLA 数据闭环流程模拟

要求：
1. DataPipeline 类，模拟从原始遥操数据 -> 清洗 -> 增强 -> Token 化
2. __init__(self, filter_threshold: float, augment_prob: float)
3. add_raw(data: dict) -> None：
   - data 格式：{"trajectory": List[float], "success": bool, "quality_score": float}
   - 加入原始队列
4. process_batch(batch_size: int) -> List[dict]：
   - 从原始队列取 batch_size 条
   - 清洗（filter）：只保留 success=True 且 quality_score >= filter_threshold 的
   - 增强（augment）：对保留的数据，以 augment_prob 概率做数据增强（trajectory 每个元素加微小噪声 np.random.normal(0, 0.01)）
   - Token 化（tokenize）：把 trajectory（List[float]）通过 ActionTokenizer 编码为 token ids
     - 为简化，内置一个简单 tokenizer：均匀分 256 bins，范围 [-1, 1]
   - 返回处理后的样本列表，每个样本为 {"tokens": List[int], "original": dict}
5. get_stats() -> dict：返回 {"total_in": int, "total_out": int, "drop_rate": float}

C++ 程序员注意：
- 原始队列用 list 或 deque
- filter 用列表推导式：[d for d in batch if d["success"] and ...]
- np.random.normal(0, 0.01, size=len(traj)) 生成噪声数组，直接加到 traj
- 注意 drop_rate = (total_in - total_out) / total_in，除零保护
"""

from typing import List, Dict
import numpy as np
import pytest


class DataPipeline:
    def __init__(self, filter_threshold: float = 0.8, augment_prob: float = 0.5):
        # TODO:
        # 1. self.raw_queue = []
        # 2. self.filter_threshold = filter_threshold
        # 3. self.augment_prob = augment_prob
        # 4. 统计量：total_in, total_out
        pass

    def _simple_tokenize(self, trajectory: List[float]) -> List[int]:
        # TODO: 内置简单 tokenizer：clip 到 [-1,1]，分 256 bins，int((x+1)/2*255)
        # 边界保护：越界值取 0 或 255
        pass

    def add_raw(self, data: dict):
        # TODO: 加入 raw_queue，total_in += 1
        pass

    def process_batch(self, batch_size: int) -> List[dict]:
        # TODO:
        # 1. 从 raw_queue 取前 batch_size 条（pop(0) 或 popleft）
        # 2. 清洗 filter
        # 3. 增强 augment（概率判断用 np.random.random() < augment_prob）
        # 4. Token 化
        # 5. total_out += len(result)
        # 6. 返回结果列表
        pass

    def get_stats(self) -> dict:
        # TODO: 返回统计字典，drop_rate 注意除零
        pass


class TestDataPipeline:
    def test_filter(self):
        pipe = DataPipeline(filter_threshold=0.9, augment_prob=0.0)
        pipe.add_raw({"trajectory": [0.1], "success": True, "quality_score": 0.5})
        pipe.add_raw({"trajectory": [0.2], "success": True, "quality_score": 0.95})

        result = pipe.process_batch(batch_size=2)
        assert len(result) == 1  # 只有 0.95 的通过
        assert result[0]["original"]["quality_score"] == 0.95

    def test_tokenize_shape(self):
        pipe = DataPipeline(filter_threshold=0.0, augment_prob=0.0)
        pipe.add_raw({"trajectory": [0.0, 0.5, -0.5], "success": True, "quality_score": 1.0})

        result = pipe.process_batch(batch_size=1)
        tokens = result[0]["tokens"]
        assert len(tokens) == 3
        assert all(0 <= t <= 255 for t in tokens)

    def test_stats(self):
        pipe = DataPipeline(filter_threshold=1.0, augment_prob=0.0)
        for _ in range(10):
            pipe.add_raw({"trajectory": [0.0], "success": True, "quality_score": 0.5})

        pipe.process_batch(batch_size=10)
        stats = pipe.get_stats()
        assert stats["total_in"] == 10
        assert stats["total_out"] == 0
        assert abs(stats["drop_rate"] - 1.0) < 1e-6

    def test_augment(self):
        pipe = DataPipeline(filter_threshold=0.0, augment_prob=1.0)
        pipe.add_raw({"trajectory": [0.0, 0.0], "success": True, "quality_score": 1.0})

        result = pipe.process_batch(batch_size=1)
        traj = result[0]["original"]["trajectory"]
        # 增强后不应完全等于 0.0
        assert not np.allclose(traj, [0.0, 0.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
