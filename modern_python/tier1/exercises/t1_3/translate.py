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


def squares(numbers: list[int]) -> list[int]:
    result = []
    for i in range(len(numbers)):
        result.append(numbers[i] * numbers[i])
    return result


def positive_evens(values: list[int]) -> list[int]:
    positives = []
    for x in values:
        if x > 0 and x % 2 == 0:
            positives.append(x)
    return positives


def name_lengths(names: list[str]) -> dict[str, int]:
    mapping = {}
    for name in names:
        mapping[name] = len(name)
    return mapping


def common_items(list_a: list, list_b: list) -> list:
    common = []
    for item in list_a:
        if item in list_b:
            common.append(item)
    return common


def is_finished(status: str) -> bool:
    if status == "paid" or status == "shipped" or status == "done":
        return True
    return False


def argmax(scores: list[int]) -> int:
    best = 0
    for i in range(len(scores)):
        if scores[i] > scores[best]:
            best = i
    return best


def print_pairs(names: list[str], scores: list[int]) -> None:
    for i in range(len(names)):
        print(f"{names[i]}: {scores[i]}")


def csv_line(fields: list[str]) -> str:
    line = ""
    for i in range(len(fields)):
        line += fields[i]
        if i != len(fields) - 1:
            line += ","
    return line


def swap(a: int, b: int) -> tuple[int, int]:
    tmp = a
    a = b
    b = tmp
    return a, b


def read_config_at(data_dir: str) -> str:
    # (10) 拼路径读文件——提示：os.path 两层嵌套，想想 pathlib 的 / 运算符
    path = os.path.join(os.path.dirname(data_dir), "..", "data", "config.toml")
    with open(path) as f:
        content = f.read()
    return content
