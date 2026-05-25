"""
【P04】Tool Parser 综合题（带嵌套 object）

要求：同 Q07，但支持嵌套 object 类型（一层嵌套即可）。
schema 示例：
{
  "type": "object",
  "properties": {
    "location": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}
  }
}
"""

import json
from typing import Dict, Tuple
import pytest


class ToolParser:
    def parse(self, json_str: str, schema: Dict) -> Tuple[Dict, str]:
        # TODO: 支持一层嵌套 object 的校验
        pass


class TestToolParser:
    def test_nested(self):
        parser = ToolParser()
        schema = {
            "type": "object",
            "properties": {
                "location": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    "required": ["x"]
                }
            },
            "required": ["location"]
        }
        args, err = parser.parse('{"location": {"x": 1.0, "y": 2.0}}', schema)
        assert err == ""
        assert args["location"]["x"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
