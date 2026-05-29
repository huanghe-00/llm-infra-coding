"""
【N24】工具调用链编排（Agent 系统工程师核心）

【描述】
自变量具身 Agent 需要根据任务类型选择正确的工具执行链：
任务类型与工具链映射：
- "pick_place" -> ["perceive", "grasp", "move", "release"]
- "open_drawer" -> ["perceive", "approach", "pull"]
- "push_button" -> ["perceive", "approach", "press"]
- "sort_objects" -> ["perceive", "classify", "grasp", "move", "release"]

给定任务类型和当前已执行的工具列表（空格分隔），
判断下一步应该执行什么工具。
如果任务类型未知，输出 UNKNOWN_TASK。
如果所有工具已执行完毕，输出 DONE。
如果当前工具序列与标准链不匹配（多执行了或顺序错了），输出 ERROR。

【输入】
第一行：task_type
第二行：已执行工具列表（空格分隔，可能为空行）

【输出】
下一步工具名称，或 DONE/UNKNOWN_TASK/ERROR

【样例输入】
pick_place
perceive grasp

【样例输出】
move

【边界】
空已执行列表 -> 输出第一个工具；
任务类型大小写敏感；
已执行列表含非法工具 -> ERROR。
"""
import sys


def solve():
    task = sys.stdin.readline().strip()
    line2 = sys.stdin.readline().strip()
    executed = line2.split() if line2 else []

    chains = {
        "pick_place": ["perceive", "grasp", "move", "release"],
        "open_drawer": ["perceive", "approach", "pull"],
        "push_button": ["perceive", "approach", "press"],
        "sort_objects": ["perceive", "classify", "grasp", "move", "release"],
    }

    if task not in chains:
        print("UNKNOWN_TASK")
        return

    chain = chains[task]

    # 校验已执行列表是否匹配前缀
    for i, tool in enumerate(executed):
        if i >= len(chain) or chain[i] != tool:
            print("ERROR")
            return

    if len(executed) >= len(chain):
        print("DONE")
    else:
        print(chain[len(executed)])


if __name__ == "__main__":
    solve()
