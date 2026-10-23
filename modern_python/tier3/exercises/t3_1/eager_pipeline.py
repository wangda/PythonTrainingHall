"""T3-1 练习 2 起始代码：一次性加载版数据管线（改写目标：生成器管线）。

数据格式：每行 "sensor-XX,value"；value 可能是空串、乱码（要过滤），
部分 value 为负数（按需求过滤掉）。
"""
import tracemalloc
from pathlib import Path


def build_source(path: Path, lines: int = 200_000) -> None:
    import random
    rng = random.Random(42)
    with path.open("w", encoding="utf-8") as f:
        for i in range(lines):
            roll = rng.random()
            if roll < 0.05:
                f.write("sensor-XX,\n")          # 空值
            elif roll < 0.10:
                f.write("sensor-XX,N/A\n")       # 乱码
            else:
                f.write(f"sensor-{i % 50:02d},{rng.uniform(-10, 100):.3f}\n")


def analyze_eager(path: Path) -> tuple[int, float]:
    rows = path.read_text(encoding="utf-8").splitlines()   # 全量进内存
    values = []
    for line in rows:
        parts = line.split(",")
        if len(parts) != 2:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if value >= 0:
            values.append(value)
    return len(values), sum(values) / len(values)


if __name__ == "__main__":
    source = Path(__file__).parent / "sensor_data.csv"
    if not source.exists():
        build_source(source)

    tracemalloc.start()
    count, mean = analyze_eager(source)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"一次性加载版: count={count}, mean={mean:.4f}, 内存峰值={peak / 1024 / 1024:.2f}MB")
