"""
【Q07】实现 Function Calling 解析器（JSON + Schema 校验）

要求：
1. 实现 ToolParser 类，parse(json_str: str, schema: dict) -> (args: dict, error: str)
2. schema 格式示例：
   {
     "type": "object",
     "properties": {
       "location": {"type": "string"},
       "force": {"type": "number"}
     },
     "required": ["location"]
   }
3. 校验规则：
   - JSON 必须可解析
   - required 字段必须存在
   - 字段类型必须匹配（只支持 string/number/integer/boolean）
   - 不匹配时返回 error 字符串，args 为空 dict
4. 不支持的字段或额外字段不报错（宽松校验）

C++ 程序员注意：
- json.loads 解析字符串，失败抛 JSONDecodeError
- dict.get(key, default) 安全访问，不要用 dict[key]（可能 KeyError）
- isinstance(val, str) 判断类型，不要用 type(val) == str
"""

import json
from typing import Dict, Tuple
import pytest


class ToolParser:
    def __init__(self):
        pass

    def parse(self, json_str: str, schema: Dict) -> Tuple[Dict, str]:
        # TODO: 实现解析与校验
        # 1. json.loads
        # 2. 检查 required
        # 3. 检查类型匹配
        # 4. 返回 (args, "") 或 ({}, error_msg)
        pass


class TestToolParser:
    def test_valid(self):
        parser = ToolParser()
        schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "force": {"type": "number"}
            },
            "required": ["location"]
        }
        args, err = parser.parse('{"location": "kitchen", "force": 0.5}', schema)
        assert err == ""
        assert args["location"] == "kitchen"

    def test_missing_required(self):
        parser = ToolParser()
        schema = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
        args, err = parser.parse('{"force": 0.5}', schema)
        assert err != ""
        assert args == {}

    def test_type_mismatch(self):
        parser = ToolParser()
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"]
        }
        args, err = parser.parse('{"count": "five"}', schema)
        assert err != ""

    def test_extra_fields_allowed(self):
        parser = ToolParser()
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"]
        }
        args, err = parser.parse('{"a": "ok", "b": 123}', schema)
        assert err == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
