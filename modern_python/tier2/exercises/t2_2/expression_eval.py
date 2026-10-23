"""T2-2 练习 1 起始代码：用 match/case 重写这个求值器。

表达式用嵌套 tuple 表示：
    ("add", 1, 2)                        -> 3
    ("mul", ("add", 1, 2), 3)            -> 9
    ("neg", ("add", 1, ("mul", 2, 5)))   -> -11
"""


def evaluate(expr):
    if isinstance(expr, (int, float)):
        return expr
    op = expr[0]
    if op == "add":
        return evaluate(expr[1]) + evaluate(expr[2])
    elif op == "mul":
        return evaluate(expr[1]) * evaluate(expr[2])
    elif op == "neg":
        return -evaluate(expr[1])
    else:
        raise ValueError(f"unknown op: {op}")


if __name__ == "__main__":
    assert evaluate(("add", 1, 2)) == 3
    assert evaluate(("mul", ("add", 1, 2), 3)) == 9
    assert evaluate(("neg", ("add", 1, ("mul", 2, 5)))) == -11
    print("all passed")
