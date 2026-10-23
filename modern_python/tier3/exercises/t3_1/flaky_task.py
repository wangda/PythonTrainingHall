"""T3-1 练习 1 起始代码：给这两个函数手写 @timer 和 @retry。

规则：先不查资料写，写完再对照 01_语言底层机制.md 核对。
"""
import random
import time

_call_count = 0


def slow_square(x: float) -> float:
    time.sleep(0.3)  # 模拟耗时操作
    return x * x


def flaky_api(value: str) -> str:
    """前两次调用必失败，第三次成功。"""
    global _call_count
    _call_count += 1
    if _call_count % 3 != 0:
        raise ConnectionError(f"attempt {_call_count}: connection refused")
    return f"ok: {value}"


if __name__ == "__main__":
    # 装饰完成后，这里应该：
    # 1. slow_square 带 @timer，打印耗时 ~0.3s
    # 2. flaky_api 带 @timer 和 @retry(times=3, delay=0.1)，打印总耗时（含两次重试等待）
    print(random.random())  # random 仅避免未使用导入告警，装饰完成后可删
    print(slow_square(4))
    try:
        print(flaky_api("hello"))
    except ConnectionError as exc:
        print(f"预期失败（未装饰 @retry）：{exc}")
