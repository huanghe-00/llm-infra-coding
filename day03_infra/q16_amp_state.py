"""
【Q16】混合精度训练状态管理（FP16+FP32 副本 + Loss Scale）

要求：
1. AMPState 类，管理 master/param/grad 的 FP16/FP32 转换
2. __init__(self, params: List[np.ndarray])：
   - params 是 FP32 的 master 权重
   - 内部维护 fp32_params（拷贝）、fp16_params（astype(np.float16)）
3. backward(fp16_grads: List[np.ndarray]) -> bool：
   - 检查所有梯度是否有 inf/nan（np.isfinite）
   - 若有：loss_scale /= 2，返回 False（跳过更新）
   - 若无：用 fp16_grads / loss_scale 更新 fp32_params（简化 SGD，lr=1）
   - 然后 fp16_params = fp32_params.astype(np.float16)
   - 连续 N 步（如 2000 步）无溢出后，loss_scale *= 2（上限 65536）
4. get_params() -> fp16_params（用于前向）

C++ 程序员注意：
- np.isfinite(x) 判断是否有 inf/nan，返回 bool 数组
- np.any(...) 判断是否有 True
- 列表拷贝用 [p.copy() for p in params]，不是 list(params)（浅拷贝问题）
"""

from typing import List
import numpy as np
import pytest


class AMPState:
    def __init__(self, params: List[np.ndarray], initial_loss_scale: float = 1024.0):
        # TODO:
        # 1. fp32_params = [p.copy() for p in params]
        # 2. fp16_params = [p.astype(np.float16) for p in params]
        # 3. loss_scale = initial_loss_scale
        # 4. 记录连续成功步数 good_steps = 0
        pass

    def backward(self, fp16_grads: List[np.ndarray]) -> bool:
        # TODO:
        # 1. 检查 fp16_grads 中是否有 inf/nan（用 np.isfinite + np.any）
        # 2. 若有：loss_scale /= 2，good_steps = 0，返回 False
        # 3. 若无：fp32_params[i] -= fp16_grads[i] / loss_scale（简化 SGD）
        #         fp16_params[i] = fp32_params[i].astype(np.float16)
        #         good_steps += 1
        #         若 good_steps >= 2000 且 loss_scale < 65536：loss_scale *= 2，good_steps = 0
        # 4. 返回 True
        pass

    def get_params(self) -> List[np.ndarray]:
        # TODO: 返回 fp16_params
        pass

    def get_loss_scale(self) -> float:
        # TODO: 返回当前 loss_scale
        pass


class TestAMPState:
    def test_basic_update(self):
        p = [np.ones((2, 2), dtype=np.float32)]
        amp = AMPState(p, initial_loss_scale=1.0)

        grads = [np.ones((2, 2), dtype=np.float16) * 0.5]
        success = amp.backward(grads)
        assert success is True
        fp16 = amp.get_params()
        assert fp16[0].dtype == np.float16
        # 更新后值应变小
        assert np.all(fp16[0] < 1.0)

    def test_inf_detection(self):
        p = [np.ones((2, 2), dtype=np.float32)]
        amp = AMPState(p, initial_loss_scale=1.0)

        grads = [np.array([[np.inf, 0], [0, 0]], dtype=np.float16)]
        success = amp.backward(grads)
        assert success is False
        # loss_scale 应下降
        assert amp.get_loss_scale() == 0.5

    def test_nan_detection(self):
        p = [np.ones((2, 2), dtype=np.float32)]
        amp = AMPState(p, initial_loss_scale=4.0)

        grads = [np.array([[np.nan, 1], [1, 1]], dtype=np.float16)]
        success = amp.backward(grads)
        assert success is False
        assert amp.get_loss_scale() == 2.0

    def test_loss_scale_growth(self):
        p = [np.ones((1,), dtype=np.float32)]
        amp = AMPState(p, initial_loss_scale=1.0)

        # 连续成功 2000 步，loss_scale 应翻倍
        for _ in range(2000):
            amp.backward([np.zeros((1,), dtype=np.float16)])

        assert amp.get_loss_scale() == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
