"""
【P07】ReAct 单步综合题（带工具注册）

要求：实现 ReActAgent，增加工具注册：
1. register_tool(name, func, schema)
2. step(user_input) 内部自动 THINK -> 选择 tool -> 执行 -> 返回结果
3. 简化：用 if/elif 根据 user_input 关键词选择工具
"""

from typing import Callable, Dict, Tuple
import pytest


class ReActAgent:
    def __init__(self):
        # TODO: tools 字典
        pass

    def register_tool(self, name: str, func: Callable, schema: dict):
        # TODO
        pass

    def step(self, user_input: str) -> Tuple[str, str]:
        # TODO: 解析输入，调用对应 tool，返回 (tool_name, result)
        pass


class TestReActStep:
    def test_tool_call(self):
        agent = ReActAgent()
        agent.register_tool("weather", lambda x: "sunny", {})

        name, result = agent.step("查天气")
        assert name == "weather"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
