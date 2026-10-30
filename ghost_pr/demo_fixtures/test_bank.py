"""Demo test fixture — intentionally weak tests for the withdraw function."""

from bank import withdraw


def test_withdraw_basic():
    assert withdraw(100, 40) == 60
