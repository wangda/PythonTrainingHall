"""T2-3 练习 3 起始代码：CPU 密集计算的朴素实现。

200 个二维点、两两距离平方和。练习目标：证明 asyncio 对这种任务无效，
并写出 NumPy 向量化版本。
"""
import time


def make_positions(n: int) -> list[tuple[float, float]]:
    return [(float(i), float(i * i % 97)) for i in range(n)]


def total_squared_distance_naive(positions: list[tuple[float, float]]) -> float:
    total = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            total += dx * dx + dy * dy
    return total


if __name__ == "__main__":
    positions = make_positions(200)
    start = time.perf_counter()
    result = total_squared_distance_naive(positions)
    elapsed = time.perf_counter() - start
    print(f"朴素版: result={result:.2f}, 耗时={elapsed * 1000:.2f}ms")
