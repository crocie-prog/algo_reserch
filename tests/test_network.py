"""Реальный API Bybit (сеть). Запуск вручную: pytest -m network."""
import pandas as pd
import pytest

from src.config import load_config
from src.data import bybit

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def ex():
    return bybit.make_exchange(load_config())


def test_klines_day_btc_1h(ex):
    s = pd.Timestamp("2023-06-01", tz="UTC")
    df = bybit.fetch_klines(ex, "BTCUSDT", "1h", s, s + pd.Timedelta(hours=23))
    assert len(df) == 24 and list(df.columns) == bybit.KLINE_COLUMNS
    assert (df["turnover"] > df["volume"]).all()                 # quote ≫ base для BTC


def test_klines_pagination_1m_crosses_pages(ex):
    s = pd.Timestamp("2023-06-01", tz="UTC")
    df = bybit.fetch_klines(ex, "BTCUSDT", "1m", s, s + pd.Timedelta(minutes=2499))
    assert len(df) == 2500 and df.index.is_unique and df.index.is_monotonic_increasing


def test_no_unclosed_bar(ex):
    now = bybit.server_time(ex)
    df = bybit.fetch_klines(ex, "ETHUSDT", "1h", now - pd.Timedelta(hours=5))
    assert df.index[-1] + pd.Timedelta(hours=1) <= now


def test_first_bars_match_config(ex):
    cfg = load_config()
    for sym, fb in cfg["universe"]["first_bars"].items():
        info = bybit.fetch_instrument_info(ex, sym)
        df = bybit.fetch_klines(ex, sym, "1m", info["launch_time"],
                                pd.Timestamp(fb, tz="UTC") + pd.Timedelta(minutes=5))
        assert df.index[0] == pd.Timestamp(fb, tz="UTC")


def test_funding_window(ex):
    since = pd.Timestamp("2023-06-01", tz="UTC")
    f = bybit.fetch_funding(ex, "SOLUSDT", since=since)
    assert f.index[0] == since + pd.Timedelta(hours=8)
    assert len(f) > 200                                           # пагинация назад сработала
    assert f.index.is_monotonic_increasing and f.index.is_unique
