"""Общие функции стратегий: окна по рабочим барам, машина состояний, описательная статистика."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import position_stats
from src.strategies._common import run_state_machine, rolling_std, sma, working_rolling


def _x(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=n).cumsum(),
                     index=pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC"))


def test_rolling_without_downtime_equals_pandas():
    x = _x()
    pd.testing.assert_series_equal(sma(x, 5), x.rolling(5).mean())
    pd.testing.assert_series_equal(rolling_std(x, 5), x.rolling(5).std(ddof=1))


def test_window_counts_working_bars_only():
    x = _x()
    down = np.zeros(30, bool)
    down[10:14] = True                                  # простой внутри окна
    m = sma(x, 5, down)
    assert m.iloc[10:14].isna().all()
    # на баре 15 окно = рабочие бары 7, 8, 9, 14, 15
    assert m.iloc[15] == pytest.approx(x.iloc[[7, 8, 9, 14, 15]].mean())
    ref = x[~down].rolling(5).std().reindex(x.index)
    pd.testing.assert_series_equal(rolling_std(x, 5, down), ref)


def test_working_rolling_validates():
    with pytest.raises(ValueError):
        working_rolling(_x(), 0, None, "mean")
    with pytest.raises(ValueError):
        working_rolling(_x(), 3, None, "median")


def _b(s):
    return np.array([c == "1" for c in s])


def test_state_machine_transitions():
    #             t: 0123456789
    el = _b("0110000000")
    es = _b("0000000100")
    xl = _b("0000110110")
    xs = _b("0000000001")
    pos = run_state_machine(el, es, xl, xs)
    # вход t1; держим t2-3; выход t4; флэт t5-6; t7: шорт-вход (флэт) ; t8 держим; t9 выход
    assert pos.tolist() == [0, 1, 1, 1, 0, 0, 0, -1, -1, 0]


def test_reversal_same_bar_and_warmup():
    el = _b("10000")
    es = _b("00100")
    xl = _b("00000")
    xs = _b("00000")
    assert run_state_machine(el, es, xl, xs).tolist() == [1, 1, -1, -1, -1]
    assert run_state_machine(el, es, xl, xs, warmup=1).tolist() == [0, 0, -1, -1, -1]


def test_frozen_holds_and_blocks():
    el = _b("10100")
    es = _b("00000")
    xl = _b("01001")
    xs = _b("00000")
    fr = _b("01100")                                    # выход t1 и вход t2 приходятся на простой
    assert run_state_machine(el, es, xl, xs, frozen=fr).tolist() == [1, 1, 1, 1, 0]


def test_conflicting_entries_stay_flat():
    assert run_state_machine(_b("1"), _b("1"), _b("0"), _b("0")).tolist() == [0]


def test_position_stats():
    idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    pos = pd.Series([0, 1, 1, -1, -1, 0, 0, 1, 1, 1], index=idx, dtype=float)
    st = position_stats(pos, periods_per_year=8760)
    assert st["n_trades"] == 3 and st["reversal_share"] == pytest.approx(1 / 3)
    assert st["exposure"] == pytest.approx(0.7)
    assert st["mean_duration_bars"] == pytest.approx(7 / 3)
    assert st["trades_per_year"] == pytest.approx(3 / (10 / 8760))
