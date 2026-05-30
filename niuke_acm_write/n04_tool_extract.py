"""
【N04】工具参数提取（JSON Dict 简化版）

【描述】
给定一个简化 JSON 字符串（Python dict 字面量格式，无嵌套 object，值只有 str/int/float），
以及 required 字段列表（空格分隔），提取所有 required 字段的值。
如果任一 required 字段缺失，输出 ERROR。
否则按 required 列表顺序输出 "key=value"，逗号分隔。

【输入】
第一行：简化 dict 字符串，如 {"location":"kitchen","force":0.5}
第二行：required 字段列表，空格分隔，如 location force

【输出】
提取结果字符串，如 location=kitchen,force=0.5
或 ERROR

【边界】
输入字符串可能单引号或双引号，用 ast.literal_eval 或 eval 安全解析；
字段缺失严格判断（大小写敏感）。
"""
import sys, ast

def solve():
    json_str = sys.stdin.readline().strip()
    if not json_str:
        return
    try:
        args = ast.literal_eval(json_str)
    except Exception:
        print("ERROR")
        return

    print(f"args:{args}", file=sys.stderr)
    required_line = sys.stdin.readline().strip()
    print(f"required_line:{required_line}", file=sys.stderr)
    required: List[str] = list(map(str, required_line.split()))
    result = []
    for key in required:
        if key not in args:
            print("ERROR")
            return
        result.append(f"{key}={args[key]}")
    print(",".join(result))  # ("分隔内容".join(可迭代对象))

if __name__ == "__main__":
    solve()
