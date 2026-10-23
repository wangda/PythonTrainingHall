"""T3-2 练习 2 起始代码：朴素滑动平均（O(n·w)），向量化目标 O(n)。
"""
import time


def moving_average_naive(values: list[float], window: int) -> list[float]:
    result = []
    for i in range(len(values) - window + 1):
        result.append(sum(values[i:i + window]) / window)
    return result


if __name__ == "__main__":
    rng = __import__("random").Random(1)
    values = [rng.gauss(0, 1) for _ in range(100_000)]

    start = time.perf_counter()
    result = moving_average_naive(values, 100)
    elapsed = time.perf_counter() - start
    print(f"朴素版(10万点,窗口100): {elapsed:.3f}s, len={len(result)}")
