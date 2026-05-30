"""
【N22】多模态输入融合（视觉 + 语言 + 力觉）

【描述】
自变量 WALL-A 模型接收三种模态输入：
- 视觉特征：dim_v 维浮点向量
- 语言指令：dim_l 维浮点向量
- 力觉反馈：dim_f 维浮点向量

系统工程师需要将它们拼接为一个统一输入向量，并插入模态类型标记：
- 视觉段前插入标记 [1, 0, 0]
- 语言段前插入标记 [0, 1, 0]
- 力觉段前插入标记 [0, 0, 1]

最终输出向量 = [1,0,0] + 视觉特征 + [0,1,0] + 语言指令 + [0,0,1] + 力觉反馈

【输入】
第一行：dim_v dim_l dim_f（整数）
第二行：dim_v 个浮点数（视觉，dim_v=0 时为空行）
第三行：dim_l 个浮点数（语言，dim_l=0 时为空行）
第四行：dim_f 个浮点数（力觉，dim_f=0 时为空行）

【输出】
一行，所有元素空格分隔（先标记，再视觉，再标记，再语言，再标记，再力觉）

【样例输入】
2 2 1
0.1 0.2
0.3 0.4
0.5

【样例输出】
1 0 0 0.1 0.2 0 1 0 0.3 0.4 0 0 1 0.5

【边界】
任意 dim 可能为 0（该模态缺失，只输出标记）；
所有值保留原始精度输出（不额外格式化）。
"""
import sys


def solve():
    dim_v, dim_l, dim_f = map(int, sys.stdin.readline().strip().split())
    nums_v: List[float] = list(map(float, sys.stdin.readline().strip().split()))
    nums_l: List[float] = list(map(float, sys.stdin.readline().strip().split()))
    nums_f: List[float] = list(map(float, sys.stdin.readline().strip().split()))
    if dim_v != len(nums_v) or dim_l != len(nums_l) or dim_f != len(nums_f):
        raise ValueError("数据输入错误")
    # python里列表直接相加就是拼接

    result: List[float] = [1, 0, 0] + nums_v + [0, 1, 0] + nums_l + [0, 0, 1] + nums_f
    print(" ".join(map(str, result)))
    print("============插桩验证已编译==========", file=sys.stderr)
    return

if __name__ == "__main__":
    solve()

# import sys


# def solve():
#     line = sys.stdin.readline().strip()
#     if not line:
#         return
#     dv, dl, df = map(int, line.split())
    
#     # 始终读三行，根据 dim 决定是否解析
#     line_v = sys.stdin.readline().strip()
#     line_l = sys.stdin.readline().strip()
#     line_f = sys.stdin.readline().strip()
    
#     v = list(map(float, line_v.split())) if dv > 0 and line_v else []
#     l = list(map(float, line_l.split())) if dl > 0 and line_l else []
#     f = list(map(float, line_f.split())) if df > 0 and line_f else []
    
#     # 构建输出：标记 + 数据 + 标记 + 数据 + 标记 + 数据
#     result = [1, 0, 0] + v + [0, 1, 0] + l + [0, 0, 1] + f
#     print(" ".join(map(str, result)))


# if __name__ == "__main__":
#     solve()