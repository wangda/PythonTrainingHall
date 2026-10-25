"""T1-3 练习 1 起始代码：10 段 Java 味代码，逐段改写成 Pythonic 版本。

任务（详见 ../03_练习.md 题 1）：
    1. 每个函数改写为行为一致的 Pythonic 版本。
    2. 每个函数上方加一行注释，说明改动为什么更 Pythonic（一句话）。
    3. 验收：ruff check（含 UP/SIM 规则组）零告警；10 处注释齐全。

提示对照：
    (4) list 的 in 是 O(n)，想想 set
    (6) 不用 max+index 两趟，想想 enumerate / max(key=)

自查：改完后全文搜索 range(len( ——应该一个都不剩。
"""
from typing import Any

def squares(numbers: list[int]) -> list[int]:
    return [n*n for n in numbers]


def positive_evens(values: list[int]) -> list[int]:
    return [x for x in values if x > 0 and x % 2 == 0]


def name_lengths(names: list[str]) -> dict[str, int]:
    return {name: len(name) for name in names}


def common_items(list_a: list[Any], list_b: list[Any]) -> list[Any]:
    return [x for x in list_a if x in list_b]


def is_finished(status: str) -> bool:
    return status == "paid" or status == "shipped" or status == "done"


def argmax(scores: list[int]) -> int:
    best: int = 0
    for i, n in enumerate(scores):
        if n > scores[best]:
            best = i
    return best


def print_pairs(names: list[str], scores: list[int]) -> None:
    for i, s in enumerate(names):
        print(f"{s}: {scores[i]}")


def csv_line(fields: list[str]) -> str:
    return ",".join(fields)


def swap(a: int, b: int) -> tuple[int, int]:
    b, a = a, b
    return a, b


def read_config_at(data_dir: str) -> str:
    # (10) 拼路径读文件——提示：os.path 两层嵌套，想想 pathlib 的 / 运算符
    from pathlib import Path
    path = Path(data_dir) / ".." / "data" / "config.toml"
    with open(path) as f:
        content = f.read()
    return content
