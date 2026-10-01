"""Шаг 5 этапа 1: загрузка и проверки funding."""
import pandas as pd

from src.data import fetch, quality, store
from tests.fakes import FakeBybit


def _f(start, n, freq="8h"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC").as_unit("ns")
    return pd.Series([1e-4 * ((i % 5) - 1) for i in range(n)], index=idx)


def test_update_funding_full_then_incremental(cfg):
    f = _f("2020-03-25 16:00", 700)
    ex = FakeBybit(f.index[499] + pd.Timedelta(minutes=1), funding={"X": f.iloc[:500]})
    assert fetch.update_funding(ex, cfg, "X") == 500
    ex2 = FakeBybit(f.index[-1] + pd.Timedelta(minutes=1), funding={"X": f})
    assert fetch.update_funding(ex2, cfg, "X") == 200
    got = store.read(cfg["paths"]["raw"], "X", "funding")["funding_rate"]
    assert got.index.equals(f.index.rename("funding_time")) or (got.index == f.index).all()
    assert (got.values == f.values).all()
    assert fetch.update_funding(ex2, cfg, "X") == 0


def test_funding_checks_interval_change_and_offgrid(cfg):
    a = _f("2023-01-01", 30)                           # 8 ч
    b = _f(a.index[-1] + pd.Timedelta(hours=4), 12, "4h")   # смена на 4 ч
    odd = pd.Series([1e-4], index=pd.DatetimeIndex([b.index[-1] + pd.Timedelta(hours=8, minutes=1)]))
    s = pd.concat([a, b, odd])
    iss, summ = quality.check_funding(s, "X", cfg)
    iv = iss[iss.check == "funding_interval"]
    assert iv.iloc[0]["n_bars"] == 12 and "240" in iv.iloc[0]["detail"]
    assert summ["funding_offgrid"] == 1
    assert "480:29" in summ["funding_intervals_min"]
