"""Шаг 3 этапа 1: инкрементальная загрузка."""
import pandas as pd
import pytest

from src.data import fetch, store
from tests.fakes import FakeBybit, make_bars

INFO = {"X": {"symbol": "X", "status": "Trading", "launchTime": "1672531200000",
              "fundingInterval": 480}}   # 2023-01-01


def _after(bars):
    """Момент, когда последний бар уже закрыт с учётом close_lag."""
    return bars.index[-1] + pd.Timedelta(minutes=61)


def _ex(bars, now):
    return FakeBybit(now, bars={("X", "60"): bars}, instruments=INFO)


def test_first_then_incremental(cfg):
    bars = make_bars("2023-01-01", 3000, "1h")
    ex = _ex(bars.iloc[:2000], bars.index[1999] + pd.Timedelta(minutes=10))
    cfg["fetch"]["chunk_bars"] = 700
    assert fetch.update_klines(ex, cfg, "X", "1h") == 1999          # последний незакрыт
    ex2 = _ex(bars, _after(bars))
    assert fetch.update_klines(ex2, cfg, "X", "1h") == 1001
    got = store.read(cfg["paths"]["raw"], "X", "1h")
    pd.testing.assert_frame_equal(got, bars, check_freq=False)
    assert fetch.update_klines(ex2, cfg, "X", "1h") == 0            # повторный запуск


def test_revision_of_last_bar_raises(cfg):
    bars = make_bars("2023-01-01", 100, "1h")
    fetch.update_klines(_ex(bars, _after(bars)), cfg, "X", "1h")
    revised = bars.copy()
    revised.iloc[-1, 3] *= 1.01
    with pytest.raises(store.RevisionError):
        fetch.update_klines(_ex(revised, _after(bars)), cfg, "X", "1h")


def test_refetch_last_overwrites_and_logs(cfg):
    bars = make_bars("2023-01-01", 100, "1h")
    fetch.update_klines(_ex(bars, _after(bars)), cfg, "X", "1h")
    revised = bars.copy()
    revised.iloc[-2:, 3] *= 1.01
    fetch.update_klines(_ex(revised, _after(bars)), cfg, "X", "1h",
                        refetch_last=5)
    got = store.read(cfg["paths"]["raw"], "X", "1h")
    pd.testing.assert_frame_equal(got, revised, check_freq=False)
    logrow = pd.read_csv(cfg["paths"]["refetch_log"]).iloc[-1]
    assert logrow["n_changed"] == 2 and logrow["n_compared"] == 5


def test_cli_parse():
    a = fetch.parse_args(["--symbols", "BTCUSDT", "--tf", "1h", "--refetch-last", "3"])
    assert a.symbols == ["BTCUSDT"] and a.tf == ["1h"] and a.refetch_last == 3
