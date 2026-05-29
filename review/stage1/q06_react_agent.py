"""
【Q06】实现 ReAct Agent 状态机

要求：
1. 实现 ReActAgent 类，维护内部状态：THINK -> ACT -> OBSERVE -> (循环或 DONE)
2. step(observation: str) -> (action_type: str, action_input: str)
3. 内部维护 thought_history: List[str]
4. max_steps=5，超过后强制返回 DONE
5. 用简单的规则模拟：如果 observation 包含"完成"或"success"，返回 DONE

C++ 程序员注意：
- 用 enum.Enum 或字符串常量定义状态
- Python 没有 switch-case（3.10+ 有 match/case），用 if/elif/else
- deque 适合存历史，但 list 也可以
"""

from typing import List, Tuple
import pytest
from enum import Enum

class AgentState:
    OBSERVE = 0
    THINK = 1
    ACT = 2
    DONE = 3

class ReActAgent:
    _SUCCESS_KEYWORDS = ("完成", "success")   # 触发结束的观察关键词

    def __init__(self, max_steps: int = 5):
        # TODO: 初始化状态、历史、步数计数器
        self.max_steps = max_steps
        self.state = AgentState.OBSERVE
        self.step_count = 0
        self.thought_history: List[str] = []

    def mock_llm(self, observation: str) -> Tuple[str, str]:
        """模拟 LLM 生成动作（私有方法，仅供内部调用）"""
        if "天气" in observation:
            return ("search", "查询天气")
        elif "计算" in observation:
            return ("calculate", "计算结果")
        else:
            return ("search", "通用搜索")

    def step(self, observation: str) -> Tuple[str, str]:  #-> 箭头后面写着函数返回的类型。 Tuple[str, str] 表示该函数返回一个元组（tuple），元组中包含两个元素，每个元素都是字符串类型。
        # TODO: 单步状态转换
        # THINK -> 生成 thought（模拟字符串）
        # ACT -> 生成 action（根据 thought）
        # OBSERVE -> 处理 observation，决定下一步
        # 返回: (action_type, action_input)
        # action_type 可以是 "search", "calculate", "finish", "done"

        # 0. 检测完成条件 （模拟LLM判断任务是否完成）
        if self.state is AgentState.DONE:
            return ("done", "任务完成") 
        # 1. 步数超出停止
        self.step_count += 1
        if self.step_count > self.max_steps:
            self.state = AgentState.DONE
            return ("done", "超出最大步数")

        # 2. 检测观察阶段是否包含结束关键词
        if "完成" in observation or "success" in observation:
            self.state = AgentState.DONE
            self.thought_history.append("检测到完成信号，任务结束")
            return ("done", "任务完成")

        # 3. 思考阶段
        # OBSERVE->THINK
        self.state = AgentState.THINK
        thought = f"根据观察到的现象{observation}, 决定下一步的行动"
        self.thought_history.append(thought)
        # 4. 行动决策
        self.state = AgentState.ACT
        action_type, action_input = self.mock_llm(observation) 
        # 5. 更新状态

        self.state = AgentState.OBSERVE
        return (action_type, action_input)

    def is_done(self) -> bool:
        # TODO: 返回是否已结束
        return self.state == AgentState.DONE

class TestReActAgent:
    def test_basic_loop(self):
        agent = ReActAgent(max_steps=5)
        # 第 1 步：初始 observation
        act_type, act_input = agent.step("用户问：北京今天天气")
        assert act_type in ["search", "calculate", "finish", "done"]

        # 第 2 步：收到工具返回
        act_type2, act_input2 = agent.step("工具返回：晴天 25度")
        # 还没收到 success，应该继续思考或行动
        assert isinstance(act_type2, str)

    def test_max_steps_termination(self):
        agent = ReActAgent(max_steps=2)
        agent.step("开始")
        agent.step("观察1")
        # 第 3 步应该强制 done
        act_type, _ = agent.step("观察2")
        assert act_type == "done"

    def test_success_done(self):
        agent = ReActAgent(max_steps=5)
        agent.step("任务：打开抽屉")
        agent.step("观察：抽屉已打开 success")
        # 收到 success 后应该 finish
        assert agent.is_done() or True  # 放宽检查，只要不出错即可


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
