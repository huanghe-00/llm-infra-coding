"""
【Q04】实现文本生成采样策略（Greedy / Top-K / Top-P / Temperature）

要求：
1. 实现 sample(logits, temperature, top_k, top_p) -> int (token_id)
2. logits: 1D numpy 数组，长度 vocab_size
3. temperature: 除法缩放（0 表示 greedy）
4. top_k: 保留概率最高的 k 个，其余置 -inf
5. top_p (nucleus): 按概率降序累加，保留最小集合使累加概率 >= top_p
6. 如果 temperature == 0，直接返回 argmax（greedy）

C++ 程序员注意：
- np.argsort 返回排序后的索引，相当于 C++ 的 sort + 索引跟踪
- np.cumsum 是前缀和
- 随机采样用 np.random.choice，传入概率分布 p
"""

import numpy as np
import pytest


def sample(logits: np.ndarray, temperature: float = 1.0,
           top_k: int = 0, top_p: float = 1.0) -> int:
    # TODO: 实现采样逻辑
    # 1. temperature 缩放（注意 temperature=0 时直接 greedy）
    # 2. top_k 过滤
    # 3. top_p (nucleus) 过滤
    # 4. softmax
    # 5. 采样返回 token_id
    """
    从 logits 中采样一个 token id。

    Args:
        logits: shape (vocab_size,)，原始模型输出分数。
        temperature: 温度系数，0 表示贪婪（不随机），>0 表示采样。
        top_k: 只从概率最高的 k 个 token 中采样，0 表示不限制。
        top_p: 核采样阈值，只保留累积概率 ≥ top_p 的最小 token 集合，1.0 表示不限制。
        注意, 是先进行topk贪心, 然后topp核采样
    Returns:
        int: 选中的 token id。
    """
    # 1. greedy 直接返回 argmax（C++ 工程师：相当于 std::max_element）
    if temperature == 0:
        token_id = int(np.argmax(logits)) # np.argmax 返回 int64，转 Python int
        return token_id

    # 2. temperature 缩放
    scaled_logits = logits / temperature

    # 3. softmax得到概率
    max_logit = np.max(scaled_logits)
    exp_logit = np.exp(scaled_logits - max_logit)
    probs = exp_logit / np.sum(scaled_logits)

    # 4. topk过滤
    if top_k > 0:
        # 找到第k大作为阈值
        if top_k >= len(probs):
            pass
        else:
            kth_value = np.partition(probs, -top_k)[-top_k]
            probs[probs < kth_value] = 0.0

    # 5. topP过滤
    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]    # np.argsort(probs)：返回升序排列后的原索引,  [::-1]：Python 的切片语法，表示反转整个数组 → 变成降序索引
        sorted_probs = probs[sorted_indices]
        # 找到最后一个需要保留的位置（累积概率刚好超过 top_p）
        # 注意：保留满足 cum_probs <= top_p 的最小集合，常见做法是包含第一个超过的 token
        # 为了稳定，保留所有 cum_probs <= top_p 的，再加上第一个超出的（如果有）
        # 更简单的标准做法：保留所有 cum_probs <= top_p 的，但如果全部低于 top_p，就全部保留。
        # 这里采用常见实现：保留 cum_probs <= top_p 的 token，如果没有则至少保留一个。
        cum_probs = np.cumsum(sorted_probs)  # 前缀和（prefix sum）
        mask = cum_probs <= top_p  # 对数组的每个元素执行比较，返回一个布尔数组
        if not np.any(mask):
            mask[0] = True

        # sorted_indices = np.array([2, 0, 1, 3])
        # mask = np.array([True, False, True, False])
        # keep = sorted_indices[mask]   # 结果: [2, 1]
        keep_indices = sorted_indices[mask]
        # 构建过滤mask
        probs_mask = np.zeros_like(probs, dtype=bool)  # np.zeros_like(probs) 创建一个和 probs 形状相同的全零数组，元素类型默认与 probs 相同（通常是 float）。
        probs_mask[keep_indices] = True
        probs[~probs_mask] = 0.0

    # 6. 重新归一化
    prob_sum = np.sum(probs)
    if prob_sum > 0.0:
        probs = probs / prob_sum
    else:
        probs = np.ones_like(probs) / len(probs)
    
    # 7. 采样
    token_id = int(np.random.choice(len(probs), p=probs))
    return token_id


class TestSampling:
    def test_greedy(self):
        logits = np.array([1.0, 2.0, 3.0, 0.5])
        token = sample(logits, temperature=0.0)
        assert token == 2  # argmax

    def test_temperature_effect(self):
        """高温使分布更均匀，低温更尖锐。这里只测不崩溃"""
        logits = np.array([1.0, 2.0, 3.0])
        for _ in range(10):
            t = sample(logits, temperature=0.5)
            assert 0 <= t < 3

    def test_top_k(self):
        logits = np.array([10.0, 1.0, 1.0, 1.0])
        # 多次采样，top_k=1 应该总是选到第一个
        for _ in range(20):
            t = sample(logits, temperature=1.0, top_k=1)
            assert t == 0

    def test_top_p(self):
        logits = np.array([5.0, 4.0, 0.1, 0.1])
        # top_p=0.5 应该只保留前两个（概率占比极高）
        results = [sample(logits, temperature=1.0, top_p=0.5) for _ in range(50)]
        assert all(r in [0, 1] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
