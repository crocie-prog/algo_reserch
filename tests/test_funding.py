"""Funding: начисление в T — с позиции, удерживаемой до T (бар, заканчивающийся в T);
на 1d — сумма начислений; положительный funding платят лонги.

Спецификация до реализации; снять skip на этапе 2.
"""
import pandas as pd
import pytest

from src.backtest.costs import funding_cost

pytestmark = pytest.mark.skip(reason="реализация funding_cost() — этап 2")


def test_hourly_attribution_and_sign():
    idx = pd.date_range("2023-01-01 00:00", periods=12, freq="1h", tz="UTC")
    # решение на close: лонг с close бара 06:00 → удерживается на барах 07:00…
    pos = pd.Series(0.0, index=idx)
    pos.loc["2023-01-01 06:00":] = 1.0
    funding = pd.Series([0.0001], index=pd.DatetimeIndex(["2023-01-01 08:00"], tz="UTC"))
    cost = funding_cost(pos, funding, idx, "1h")
    # начисление 08:00 относится к бару 07:00–08:00 (open 07:00), позиция на нём = pos[06:00] = 1
    assert cost.loc["2023-01-01 07:00"] == pytest.approx(0.0001)
    assert cost.drop(pd.Timestamp("2023-01-01 07:00", tz="UTC")).eq(0).all()


def test_short_receives():
    idx = pd.date_range("2023-01-01 00:00", periods=12, freq="1h", tz="UTC")
    pos = pd.Series(-1.0, index=idx)
    funding = pd.Series([0.0002], index=pd.DatetimeIndex(["2023-01-01 08:00"], tz="UTC"))
    cost = funding_cost(pos, funding, idx, "1h")
    assert cost.loc["2023-01-01 07:00"] == pytest.approx(-0.0002)


def test_position_opened_at_T_does_not_pay():
    idx = pd.date_range("2023-01-01 00:00", periods=12, freq="1h", tz="UTC")
    pos = pd.Series(0.0, index=idx)
    pos.loc["2023-01-01 07:00":] = 1.0   # вход на close бара 07:00 = в момент 08:00
    funding = pd.Series([0.0001], index=pd.DatetimeIndex(["2023-01-01 08:00"], tz="UTC"))
    cost = funding_cost(pos, funding, idx, "1h")
    assert cost.eq(0).all()


def test_daily_sum_of_three():
    idx = pd.date_range("2023-01-01", periods=4, freq="1D", tz="UTC")
    pos = pd.Series(1.0, index=idx)
    ts = pd.DatetimeIndex(["2023-01-02 08:00", "2023-01-02 16:00", "2023-01-03 00:00",
                           "2023-01-03 08:00"], tz="UTC")
    funding = pd.Series([0.0001, 0.0002, 0.0003, 0.0004], index=ts)
    cost = funding_cost(pos, funding, idx, "1d")
    # бар open 2023-01-02 покрывает (01-02 00:00, 01-03 00:00]: три начисления
    assert cost.loc["2023-01-02"] == pytest.approx(0.0006)
    assert cost.loc["2023-01-03"] == pytest.approx(0.0004)
