"""Детекторы look-ahead."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.checks import LookAheadError, shift_check, truncation_check
from tests.fakes import make_bars


def honest(df, w=20):
    """Каузальная стратегия: знак отклонения от скользящей средней."""
    ma = df["close"].rolling(w).mean()
    return np.sign(df["close"] - ma).fillna(0.0)


def leaky_shift(df, w=20):
    """Утечка: сигнал на t смотрит на close_{t+1}."""
    return np.sign(df["close"].shift(-1) - df["close"]).fillna(0.0)


def leaky_global(df):
    """Утечка: нормировка по всей выборке (порог — медиана всего ряда)."""
    return np.sign(df["close"] - df["close"].median()).astype("float64")


@pytest.fixture
def df():
    return make_bars("2023-01-01", 2000, "1h", seed=11)


def test_honest_passes(df):
    truncation_check(honest, df, {"w": 20})


@pytest.mark.parametrize("fn", [leaky_shift, leaky_global])
def test_leaks_caught(df, fn):
    with pytest.raises(LookAheadError):
        truncation_check(fn, df)


def test_shift_check_reports(df):
    r = shift_check(df, honest(df), periods_per_year=8760)
    assert set(r) == {"sharpe", "sharpe_lead", "delta"}
