"""T1-3 练习 3 起始代码：os.path 版目录统计，用 pathlib 重写。

任务（详见 ../03_练习.md 题 3）：
    1. 用 pathlib.Path 重写：rglob、read_text、splitlines。
    2. 原版 f.readlines() 把整个文件载入再数行——你的版本顺带修掉这个内存问题
       （想想逐块读或 str.count）。
    3. 对 modern_python/tier1/ 目录跑一遍，打印结果，与原版输出对比。

验收：行为与原版一致（行数统计口径在注释中说明）；零 os.path；ruff check 零告警。
"""

import os


def report(root):
    py_count = 0
    total_lines = 0
    for dirpath, dirnames, filenames in os.walk(root):
        print(dirpath, dirnames, filenames)
        for filename in filenames:
            if filename.endswith(".md"):
                full = os.path.join(dirpath, filename)
                py_count += 1
                with open(full, encoding="utf-8") as f:
                    total_lines += len(f.readlines())
    return py_count, total_lines


if __name__ == "__main__":
    print(report("modern_python/tier1"))
