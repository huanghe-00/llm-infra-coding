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

EPS = 1e-9

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
    # 1. 处理温度0情况，注意浮点数的处理方式
    if temperature < EPS:
        token_id = int(np.argmax(logits))
        return token_id

    # 2. 温度缩放 + softmax
    scaled = logits / temperature
    l_max = np.max(scaled)
    e = np.exp(scaled - l_max)
    e_sum = np.sum(e)
    probs = e / e_sum

    # 3. 筛选topK
    if top_k < probs.size and top_k > 0:
        sorted_indices = np.argsort(probs)
        top_k_indicies = sorted_indices[-top_k:]  # 升序，取倒数topk个
        mask = np.zeros_like(probs, dtype=bool)
        mask[top_k_indicies] = True
        probs[~mask] = 0.0

        #重新归一化
        p_sum = np.sum(probs)
        if p_sum > EPS:
            probs = probs / p_sum
        else:
            return int(np.argmax(logits))

    # 4. 过滤topP  核心函数，cumsum
    if top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]  # 升序重切片为降序
        sorted_probs = probs[sorted_idx]
        cum_probs = np.cumsum(sorted_probs)

        cutoff = int(np.argmax(cum_probs >= top_p))  # 累加和的队列里，找到最右边（最小）的满足大于top_p的idx， 注意这里有效的原因是

        if cutoff == 0 and sorted_probs[0] > top_p:
            cutoff = 1  # 返回前两个

        keep_indices = sorted_idx[:cutoff + 1]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep_indices] = True
        probs[~mask] = 0.0

        # 重新归一化
        p_sum = np.sum(probs)
        if p_sum > EPS:
            probs = probs / p_sum
        else:
            return int(np.argmax(logits)) 

    choice_token = np.random.choice(len(probs), p = probs)
    return int(choice_token)


    # 1. greedy 直接返回 argmax（C++ 工程师：相当于 std::max_element）


    # 2. temperature 缩放

    # 3. softmax得到概率


    # 4. topk过滤

    # 5. topP过滤
    

    # 6. 重新归一化

    # 7. 采样



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
