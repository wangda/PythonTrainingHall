"""T2-4 练习 2 起始代码：5 个几乎一样的测试，parametrize 重构目标。

先跑绿：uv run pytest modern_python/tier2/exercises/t2_4/test_score.py
"""
import pytest

from score_lib import normalize_score


def test_plain_number():
    assert normalize_score("90") == 90.0


def test_number_with_spaces():
    assert normalize_score(" 85 ") == 85.0


def test_out_of_range():
    assert normalize_score("150") is None


def test_garbage():
    assert normalize_score("abc") is None


def test_empty():
    assert normalize_score("") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
