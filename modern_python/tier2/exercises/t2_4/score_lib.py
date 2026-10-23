"""T2-4 练习 2 被测模块：分数解析。

行为即契约：合法 [0,100] 返回 float，其他一律 None。
"""

def normalize_score(raw: str) -> float | None:
    text = raw.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not 0 <= value <= 100:
        return None
    return value
