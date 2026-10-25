"""T1-3 练习 2 起始代码：LBYL 改写成 EAFP。

任务（详见 ../03_练习.md 题 2）：
    1. 每个函数给出 EAFP 或标准库 API 版本（保持函数名与行为，改实现）。
       提示：try/except FileNotFoundError、os.environ.get、try/except ValueError。
    2. parse_port 的 EAFP 版本必须修复一个原版潜在 bug：
       raw = "+8080" 或 "8080 " 时，原版行为是什么？你的版本行为是什么？
       分析写进函数内注释。
    3. 文件末尾注释块：举一个"这个项目里" LBYL 更合适的真实场景，说明理由。

验收：三个函数改写 + 两处行为分析注释；ruff check 零告警。
"""

import json
import os


def read_config(path):
    config = {}
    if os.path.exists(path):
        with open(path) as f:
            config = json.load(f)
    return config


def get_env(name):
    if name in os.environ:
        return os.environ[name]
    return None


def parse_port(raw):
    if raw.isdigit():
        return int(raw)
    return 8080
