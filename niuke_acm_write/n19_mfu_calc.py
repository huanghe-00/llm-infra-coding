"""
【N19】MFU（Model FLOPs Utilization）计算

【描述】
给定实测吞吐（tokens/s）、模型参数量（直接数字）、单卡峰值算力（TFLOPS）、GPU 数量。
MFU = (measured_tps * 6 * params) / (gpu_peak * 1e12 * gpu_num) * 100
输出百分比（保留2位小数）。

【输入】
一行4个值：measured_tps params gpu_peak_tflops gpu_num
（浮点数 浮点数 浮点数 整数）

【输出】
MFU 百分比（保留2位小数）

【样例输入】
1000 10000000000 312 8

【样例输出】
2.40

【边界】
极小值、极大值；除零保护（gpu_num=0 或 gpu_peak=0 输出 0.00）。
"""
import sys


def solve():
    line = sys.stdin.readline().strip()
    if not line:
        return
    parts = line.split()
    tps = float(parts[0])
    params = float(parts[1])
    peak = float(parts[2])
    num = int(parts[3])
    
    if peak == 0 or num == 0:
        print("0.00")
        return
    
    mfu = (tps * 6.0 * params) / (peak * 1e12 * num) * 100.0
    print(f"{mfu:.2f}")


if __name__ == "__main__":
    solve()