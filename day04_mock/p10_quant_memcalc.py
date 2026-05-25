"""
【P10】量化内存计算器综合题

要求：实现函数，输入模型参数和量化配置，输出模型体积和 KV Cache 体积。
config = {
    "params": 10_000_000_000,  # 10B
    "bits": 8,
    "batch": 4,
    "seq_len": 2048,
    "layers": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "gqa_group": 4  # 8 kv heads 对应 32 q heads，group=4
}
"""

import pytest


def calculate_model_size(config: dict) -> int:
    # TODO: 模型体积（bytes）= params * bits / 8
    pass


def calculate_kv_cache_size(config: dict) -> int:
    # TODO: KV Cache 体积（bytes）
    # 2(K+V) * batch * seq_len * layers * num_kv_heads * head_dim * bits / 8
    pass


class TestQuantMemCalc:
    def test_10b_fp16(self):
        config = {
            "params": 10_000_000_000,
            "bits": 16,
            "batch": 1, "seq_len": 1, "layers": 1,
            "num_kv_heads": 1, "head_dim": 1, "gqa_group": 1
        }
        assert calculate_model_size(config) == 20_000_000_000  # 10B * 2 bytes

    def test_kv_fp16(self):
        config = {
            "params": 1, "bits": 16,
            "batch": 4, "seq_len": 2048, "layers": 32,
            "num_kv_heads": 8, "head_dim": 128, "gqa_group": 4
        }
        # 2 * 4 * 2048 * 32 * 8 * 128 * 2 bytes
        expected = 2 * 4 * 2048 * 32 * 8 * 128 * 2
        assert calculate_kv_cache_size(config) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
