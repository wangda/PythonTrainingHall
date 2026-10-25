"""T1-2 练习 2 靶子代码：故意包含至少 4 类问题，供 ruff 首检。

规则：本文件代码本身就是"病灶"，你的任务是练习 02_练习.md 题 2 的流程——
ruff check 记录规则编码 → --fix 自动修 → 手动修剩下的 → format → 清零。
不要一开始就把它改"好"，否则练习作废。

练习中要记录的问题（写进笔记，不是写在这里）：
    - 每条告警的规则编码与规则名（uv run ruff rule <编码>）
    - 哪些被 --fix 自动修复、哪些必须人工判断，为什么
    - line-length 改成 40 后新触发了哪些告警

注意：import os / import sys 中 sys 是未使用导入——这也是靶子的一部分。


import os
import sys
from typing import List, Optional

def process(items=[]):
    result = []
    if len(items) == 0:
        return result
    if items[0] == None:
        return result
    tmp = items[0]
    unused = "leftover"
    return [str(x) for x in items if x != None]
"""

from typing import Any


def process(items: list[Any] | None = None) -> list[str]:
    if items is None:
        items = []

    if len(items) == 0:
        return []
    if items[0] is None:
        return []
    return [str(x) for x in items if x is not None]
