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
from typing import Dict, Tuple, Any
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

        # json字符串解析->dict  dict压缩用json.dump -> json str
        try:
            args: dict[str, any] = json.loads(json_str)
        except json.JSONDecodeError:
            return ({}, "json解析失败, {json_str}")

        # schema字段解析
        properties: dict[str, dict[str, str]] = schema.get("properties", {})
        required: list[str] = schema.get("required", [])

        # 检查必须字段
        for field in required:
            if field not in args:
                return ({}, f"缺少必要参数字段{field}")

        # 检查properties定义参数，是否有已有args突破类型约束
        for field, prop_info in properties.items():
            # field 参数名 str
            if field not in args:
                continue  # 非强校验 schema没有约束的arg不处理
            value: any = args[field]    # 拿到参数值
            expect_type: str = prop_info["type"]    # 拿到参数类型
            if not self._check_param_valid(value, expect_type):
                return ({}, f"参数{field}, 不符合类型约束")

        # 校验成功，返回结果
        return (args, "")

    def _check_param_valid(self, arg: any, expect_type: str) -> bool:
        if expect_type == "string":
            return isinstance(arg, str)
        if expect_type == "number":
            return isinstance(arg, (int, float)) and not isinstance(arg, bool)
        if expect_type == "integer":
            return isinstance(arg, int) and not isinstance(arg, bool)
        if expect_type == "boolean":
            return isinstance(arg, bool)
        return False

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
