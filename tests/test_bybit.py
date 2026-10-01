"""Шаг 1 этапа 1: запросы к Bybit (офлайн, фейковая биржа)."""
import ccxt
import pandas as pd
import pytest

from src.data import bybit
from tests.fakes import FakeBybit, make_bars



def test_last_closed_open():
    now = pd.Timestamp("2026-10-01 19:53:00", tz="UTC")
    assert bybit.last_closed_open(now, "1h", 5) == pd.Timestamp("2026-10-01 18:00", tz="UTC")
    assert bybit.last_closed_open(now, "1d", 5) == pd.Timestamp("2026-09-30", tz="UTC")
    # граница: бар 19:00 закрылся в 20:00, но lag ещё не прошёл
    edge = pd.Timestamp("2026-10-01 20:00:03", tz="UTC")
    assert bybit.last_closed_open(edge, "1h", 5) == pd.Timestamp("2026-10-01 18:00", tz="UTC")
    after = pd.Timestamp("2026-10-01 20:00:05", tz="UTC")
    assert bybit.last_closed_open(after, "1h", 5) == pd.Timestamp("2026-10-01 19:00", tz="UTC")


def test_pagination_full_history_and_unclosed_dropped():
    bars = make_bars("2023-01-01", 2500, "1h")
    now = bars.index[-1] + pd.Timedelta(minutes=30)       # последний бар незакрыт
    ex = FakeBybit(now, bars={("X", "60"): bars})
    got = bybit.fetch_klines(ex, "X", "1h", pd.Timestamp("2022-12-01", tz="UTC"), limit=1000)
    pd.testing.assert_frame_equal(got, bars.iloc[:-1], check_freq=False)
    kl = [c for c in ex.calls if c[0] == "kline"]
    for _, p in kl:                                       # окно не больше limit баров
        assert (p["end"] - p["start"]) // 3_600_000 + 1 <= 1000


def test_empty_pages_before_listing_skipped():
    bars = make_bars("2020-03-25 10:00", 50, "1h")
    ex = FakeBybit(bars.index[-1] + pd.Timedelta(hours=2), bars={("X", "60"): bars})
    got = bybit.fetch_klines(ex, "X", "1h", pd.Timestamp("2020-01-01", tz="UTC"), limit=100)
    assert len(got) == 50


def test_internal_gap_preserved_not_filled():
    bars = make_bars("2023-01-01", 300, "1h")
    gappy = bars.drop(bars.index[100:110])
    ex = FakeBybit(bars.index[-1] + pd.Timedelta(hours=2), bars={("X", "60"): gappy})
    got = bybit.fetch_klines(ex, "X", "1h", bars.index[0], limit=50)
    assert len(got) == 290
    assert got.index.equals(gappy.index)


def test_end_bound_inclusive():
    bars = make_bars("2023-01-01", 100, "1h")
    ex = FakeBybit(bars.index[-1] + pd.Timedelta(hours=2), bars={("X", "60"): bars})
    got = bybit.fetch_klines(ex, "X", "1h", bars.index[10], bars.index[20])
    assert got.index[0] == bars.index[10] and got.index[-1] == bars.index[20]


def test_retries_on_network_error():
    bars = make_bars("2023-01-01", 10, "1h")
    ex = FakeBybit(bars.index[-1] + pd.Timedelta(hours=2), bars={("X", "60"): bars}, fail_first=2)
    got = bybit.fetch_klines(ex, "X", "1h", bars.index[0], retries=3)
    assert len(got) == 10


def test_retries_exhausted_raises():
    bars = make_bars("2023-01-01", 10, "1h")
    ex = FakeBybit(bars.index[-1] + pd.Timedelta(hours=2), bars={("X", "60"): bars}, fail_first=5)
    with pytest.raises(ccxt.NetworkError):
        bybit.fetch_klines(ex, "X", "1h", bars.index[0], retries=2)


def test_off_grid_timestamp_rejected():
    bad = make_bars("2023-01-01 00:00:30", 5, "1h")
    ex = FakeBybit(bad.index[-1] + pd.Timedelta(hours=2), bars={("X", "60"): bad})
    with pytest.raises(bybit.DataError):
        bybit.fetch_klines(ex, "X", "1h", pd.Timestamp("2023-01-01", tz="UTC"))


def _funding(start, n, freq="8h"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.Series([1e-4 * (i % 7 - 3) for i in range(n)], index=idx)


def test_funding_backward_pagination_full():
    f = _funding("2020-03-25 16:00", 1000)
    ex = FakeBybit(f.index[-1] + pd.Timedelta(hours=1), funding={"X": f})
    got = bybit.fetch_funding(ex, "X", limit=200)
    assert got.index.equals(pd.DatetimeIndex(f.index, name="funding_time"))
    assert (got.values == f.values).all()


def test_funding_incremental_since():
    f = _funding("2020-03-25 16:00", 1000)
    ex = FakeBybit(f.index[-1] + pd.Timedelta(hours=1), funding={"X": f})
    got = bybit.fetch_funding(ex, "X", since=f.index[549], limit=200)
    assert got.index[0] == f.index[550] and len(got) == 450


def test_instrument_info():
    ex = FakeBybit("2026-10-01", instruments={"X": {"symbol": "X", "status": "Trading",
                                                    "launchTime": "1584230400000",
                                                    "fundingInterval": 480}})
    info = bybit.fetch_instrument_info(ex, "X")
    assert info["launch_time"] == pd.Timestamp("2020-03-15", tz="UTC")
    assert info["funding_interval_minutes"] == 480
