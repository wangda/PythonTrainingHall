"""T2-2 练习 2 起始代码：三段"同一表达式写两次"，用海象运算符消除。

要求：行为完全一致；第三段改写前后计时对比。
"""
import io
import time


# 段 1：读流循环，read 和判空写了两遍逻辑
def consume_stream(stream) -> list[str]:
    chunks = []
    while True:
        chunk = stream.read(8)
        if not chunk:
            break
        chunks.append(chunk.upper())
    return chunks


# 段 2：len 算了两遍
def warn_if_big(items) -> str | None:
    if len(items) > 3:
        return f"too many: {len(items)}"
    return None


# 段 3：expensive_normalize 对每个元素算了两遍
def expensive_normalize(raw: str) -> str:
    time.sleep(0.001)  # 模拟昂贵调用（正则/解码/反序列化）
    return raw.strip().lower()


def collect_valid(data: list[str], min_len: int) -> list[str]:
    return [expensive_normalize(x) for x in data
            if len(expensive_normalize(x)) >= min_len]


if __name__ == "__main__":
    stream = io.StringIO("abcdef gh")
    assert consume_stream(stream) == ["ABCDEF G", "H"]

    assert warn_if_big([1, 2]) is None
    assert warn_if_big([1, 2, 3, 4]) == "too many: 4"

    data = ["  Apple ", "Kiwi", "  DragonFruit "] * 10
    assert collect_valid(data, 4) == ["apple", "kiwi", "dragonfruit"] * 10

    start = time.perf_counter()
    collect_valid(data, 4)
    print(f"改写前耗时: {time.perf_counter() - start:.3f}s")
