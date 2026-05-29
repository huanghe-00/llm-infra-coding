#!/usr/bin/env python3
"""
ACM 模式测试框架 · 自变量 Agent 系统工程师笔试冲刺

用法：
    python test_runner.py n01                    # 测试单题
    python test_runner.py n01 n02 n03            # 测试多题
    python test_runner.py --all                  # 测试全部 25 题
    python test_runner.py --tier S               # 测试 Tier S（死保题）
    python test_runner.py --tier A               # 测试 Tier A（高概率题）
    python test_runner.py --list                 # 列出所有题目和用例数
    python test_runner.py n01 --quiet            # 只显示失败用例的详细输出

特性：
    - 自动匹配 testcases/{题号}/ 下的 .in / .out 文件
    - 自动查找 n01_*.py 格式的题目文件
    - 清晰区分 stdout（判题依据）和 stderr（调试输出）
    - 失败时自动显示 unified diff（预期 vs 实际）
    - 支持尾部空白容错（牛客网通常忽略行尾空格）
    - 不依赖绝对路径，脚本与测试用例同级别就能正常工作
"""

import subprocess
import sys
import os
import glob
import difflib

# ── 终端颜色 ──
class C:
    PASS = "\033[92m"      # 绿色
    FAIL = "\033[91m"      # 红色
    WARN = "\033[93m"      # 黄色
    INFO = "\033[96m"      # 青色
    BOLD = "\033[1m"
    DIM  = "\033[2m"
    RST  = "\033[0m"

def c(text, color):
    return f"{color}{text}{C.RST}"

# ── 基于脚本位置的相对路径 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def script_path(*parts):
    """拼接相对于脚本目录的路径"""
    return os.path.join(SCRIPT_DIR, *parts)

# ── Tier 定义（与 STUDY_PLAN 对应） ──
TIERS = {
    "S": ["n01","n02","n03","n04","n10","n11","n21","n22","n23","n24","n25"],
    "A": ["n05","n06","n07","n08","n09","n12","n13","n14","n17","n18","n19","n20"],
    "B": ["n15","n16"],
    "C": [],
}

# ── 辅助函数 ──
def find_py(problem_id):
    """根据题号查找对应的 .py 文件（如 n01_kv_cache_mem.py）"""
    # 优先匹配 n01_*.py 格式
    pattern = script_path(f"{problem_id}_*.py")
    candidates = glob.glob(pattern)
    if candidates:
        return candidates[0]
    # 精确匹配 n01.py
    exact = script_path(f"{problem_id}.py")
    return exact if os.path.exists(exact) else None

def run(py_file, in_file, out_file):
    """运行单个用例，返回结果字典"""
    with open(in_file, "r") as f:
        stdin = f.read()

    proc = subprocess.run(
        [sys.executable, py_file],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=10,
    )

    with open(out_file, "r") as f:
        expected = f.read()

    stdout = proc.stdout
    stderr = proc.stderr

    # 标准化比对：逐行去除尾部空白，允许最后一行有无换行的差异
    def normalize(s):
        lines = s.splitlines()
        return "\n".join(line.rstrip() for line in lines)

    passed = (normalize(expected) == normalize(stdout)) and (proc.returncode == 0)

    return {
        "pass": passed,
        "expected": expected,
        "stdout": stdout,
        "stderr": stderr,
        "rc": proc.returncode,
    }

def show_diff(expected, actual):
    """打印 unified diff"""
    exp_lines = expected.splitlines(keepends=True)
    act_lines = actual.splitlines(keepends=True)

    # 标准化最后一行换行
    if exp_lines and not exp_lines[-1].endswith("\n"):
        exp_lines[-1] += "\n"
    if act_lines and not act_lines[-1].endswith("\n"):
        act_lines[-1] += "\n"

    diff = list(difflib.unified_diff(
        exp_lines, act_lines,
        fromfile="expected", tofile="actual",
        lineterm=""
    ))

    if not diff:
        return

    print(f"    {c('差异对比:', C.WARN)}")
    for line in diff[2:]:
        if line.startswith("+"):
            print(f"      {c(line, C.FAIL)}")
        elif line.startswith("-"):
            print(f"      {c(line, C.INFO)}")
        else:
            print(f"      {c(line, C.DIM)}")

def print_case(name, res, verbose=True):
    """打印单个用例结果"""
    status = c("✓ PASS", C.PASS) if res["pass"] else c("✗ FAIL", C.FAIL)
    print(f"  [{status}] {name}")

    if not res["pass"] and verbose:
        show_diff(res["expected"], res["stdout"])

        # 简短展示 stdout
        out = res["stdout"].strip()
        if out:
            lines = out.splitlines()
            print(f"    {c('实际输出 (stdout):', C.DIM)}")
            for line in lines[:5]:
                print(f"      │ {line}")
            if len(lines) > 5:
                print(f"      │ ... ({len(lines)} 行)")
        else:
            print(f"    {c('实际输出为空', C.WARN)}")

    # stderr（调试输出）—— 无论是否通过都显示，但只显示前 3 行
    err = res["stderr"].strip()
    if err:
        print(f"    {c('调试输出 (stderr):', C.WARN)}")
        for line in err.splitlines()[:3]:
            print(f"      │ {line}")
        if len(err.splitlines()) > 3:
            print(f"      │ ... ({len(err.splitlines())} 行)")

def test_one(pid, verbose=True):
    """测试单个题目"""
    tc_dir = script_path("testcases", pid)
    if not os.path.isdir(tc_dir):
        print(c(f"✗ 未找到 {pid} 的测试用例目录: {tc_dir}", C.FAIL))
        return 0, 0

    py_file = find_py(pid)
    if not py_file:
        print(c(f"✗ 未找到 {pid} 的 Python 文件（期望 {pid}_*.py 或 {pid}.py）", C.FAIL))
        return 0, 0

    # 收集所有 .in 文件
    cases = sorted(glob.glob(os.path.join(tc_dir, "*.in")))
    if not cases:
        print(c(f"✗ {pid} 没有测试用例", C.FAIL))
        return 0, 0

    print(f"\n{c('═'*56, C.BOLD)}")
    print(f"  {c('题目:', C.BOLD)} {c(pid, C.INFO)}")
    print(f"  {c('代码:', C.DIM)} {os.path.relpath(py_file, SCRIPT_DIR)}")
    print(f"  {c('用例:', C.DIM)} {len(cases)} 组")
    print(f"{c('─'*56, C.DIM)}")

    passed = failed = 0
    for cin in cases:
        cout = cin[:-3] + ".out"  # 替换 .in 为 .out
        if not os.path.exists(cout):
            print(f"  [{c('SKIP', C.WARN)}] {os.path.basename(cin)[:-3]} (缺少 .out)")
            continue

        res = run(py_file, cin, cout)
        print_case(os.path.basename(cin)[:-3], res, verbose=verbose)

        if res["pass"]:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"{c('─'*56, C.DIM)}")
    if failed == 0:
        print(f"  结果: {c(passed, C.PASS)}/{total} 通过  {c('全部通过!', C.PASS)}")
    else:
        print(f"  结果: {c(passed, C.PASS)}/{total} 通过, {c(failed, C.FAIL)} 失败")

    return passed, failed

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ACM 模式测试框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_runner.py n01                    # 测试单题
  python test_runner.py n01 n02 --quiet        # 测试两题，静默模式
  python test_runner.py --tier S             # 测试 Tier S（死保题）
  python test_runner.py --all                # 测试全部 25 题
  python test_runner.py --list               # 查看题目清单
        """
    )
    parser.add_argument("problems", nargs="*", help="题号，如 n01 n02")
    parser.add_argument("--all", action="store_true", help="测试全部题目")
    parser.add_argument("--tier", choices=["S","A","B","C"], help="按 Tier 测试")
    parser.add_argument("--list", action="store_true", help="列出所有题目")
    parser.add_argument("--quiet", "-q", action="store_true", help="仅显示失败用例的详细输出")

    args = parser.parse_args()

    if args.list:
        print(c("题目清单与 Tier 分布:", C.BOLD))
        for tier, pids in TIERS.items():
            if pids:
                counts = []
                for p in pids:
                    n = len(glob.glob(script_path("testcases", p, "*.in")))
                    counts.append(f"{p}({n})")
                print(f"\n  Tier {c(tier, C.BOLD)}: {', '.join(counts)}")
        return

    pids = []
    if args.all:
        pids = sorted(set(sum(TIERS.values(), [])))
    elif args.tier:
        pids = TIERS.get(args.tier, [])
    elif args.problems:
        pids = args.problems
    else:
        parser.print_help()
        return

    total_pass = total_fail = 0
    for pid in pids:
        p, f = test_one(pid, verbose=not args.quiet)
        total_pass += p
        total_fail += f

    # 全局总结
    print(f"\n{c('═'*56, C.BOLD)}")
    print(f"{c('  全局总结', C.BOLD)}")
    print(f"    通过: {c(total_pass, C.PASS)}")
    print(f"    失败: {c(total_fail, C.FAIL) if total_fail else '0'}")
    print(f"    总计: {total_pass + total_fail}")

    if total_fail == 0:
        print(f"\n  {c('🎉 所有题目全部通过!', C.PASS)}")
    else:
        print(f"\n  {c('⚠ 存在失败用例，请检查上方输出', C.WARN)}")
        sys.exit(1)

if __name__ == "__main__":
    main()