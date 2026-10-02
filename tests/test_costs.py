"""Комиссия и проскальзывание."""
import pandas as pd
import pytest

from src.backtest.costs import fee_cost, turnover


def _s(vals):
    return pd.Series(vals, index=pd.date_range("2023-01-01", periods=len(vals), freq="1h", tz="UTC"),
                     dtype="float64")


def test_initial_entry_charged():
    assert fee_cost(_s([1, 1, 1]), 0.001).tolist() == pytest.approx([0.001, 0, 0])


def test_reversal_is_two_turnovers():
    assert turnover(_s([1, -1, -1, 0])).tolist() == [1, 2, 0, 1]
    assert fee_cost(_s([1, -1]), 0.001).iloc[1] == pytest.approx(0.002)


def test_fractional_and_initial_pos():
    assert turnover(_s([0.3, 0.7, -0.1]), initial_pos=0.5).tolist() == pytest.approx([0.2, 0.4, 0.8])
