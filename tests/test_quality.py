"""Шаг 4 этапа 1: слой clean и проверки качества."""
import numpy as np
import pandas as pd

from src.data import clean, quality, store
from tests.fakes import make_bars


def _write_raw(cfg, symbol, tf, df):
    store.write(cfg["paths"]["raw"], symbol, tf, df)


def _ohlc_from_1m(m1, tf):
    return quality.aggregate(m1, tf).drop(columns="n")


def test_drop_incomplete_first_bar(cfg):
    # торги начались в 10:36: первые 1h (10:00) и 1d (00:00) корзины неполные
    m1 = make_bars("2020-03-25 10:36", 3 * 1440, "1min")
    _write_raw(cfg, "BTC", "1m", m1)
    _write_raw(cfg, "BTC", "1h", _ohlc_from_1m(m1, "1h"))
    info_m = clean.build_clean(cfg, "BTC", "1m")
    info_h = clean.build_clean(cfg, "BTC", "1h")
    info_d = clean.build_clean(cfg, "BTC", "1d")
    assert info_m["n_dropped"] == 0
    # неполные: первая корзина и последняя (ряд обрывается в 10:35)
    assert info_h["dropped"][0] == pd.Timestamp("2020-03-25 10:00", tz="UTC")
    assert info_h["dropped"][-1] == pd.Timestamp("2020-03-28 10:00", tz="UTC")
    assert info_d["dropped"] == [pd.Timestamp("2020-03-25", tz="UTC"),
                                 pd.Timestamp("2020-03-28", tz="UTC")]
    got = store.read(cfg["paths"]["clean"], "BTC", "1h")
    assert got.index[0] == pd.Timestamp("2020-03-25 11:00", tz="UTC")
    # raw не тронут
    assert store.read(cfg["paths"]["raw"], "BTC", "1h").index[0] == pd.Timestamp("2020-03-25 10:00", tz="UTC")


def test_aligned_start_keeps_first_bar(cfg):
    m1 = make_bars("2021-03-15", 1440, "1min")
    _write_raw(cfg, "ETH", "1m", m1)
    clean.build_clean(cfg, "ETH", "1m")
    assert clean.build_clean(cfg, "ETH", "1h")["n_dropped"] == 0


def test_check_bars_finds_problems(cfg):
    df = make_bars("2023-01-01", 2000, "1h")
    df = df.drop(df.index[100:105]).drop(df.index[500:502])        # пропуски 5 и 2 бара
    df.loc[df.index[800:803], "volume"] = 0.0                       # нулевой объём 3 бара
    df.loc[df.index[900], "high"] = df["low"].iloc[900] * 0.9       # high < low
    t = df.index[1500]
    df.loc[t:, ["open", "high", "low", "close"]] *= 1.5             # скачок цены ≫ 10σ
    df.loc[t, "open"] = df.loc[df.index[1499], "close"]
    df.loc[t, "low"] = min(df.loc[t, "low"], df.loc[t, "open"])
    iss, s = quality.check_bars(df, "X", "1h", cfg)
    assert s["n_gaps"] == 2 and s["missing_bars"] == 7 and s["longest_gap_bars"] == 5
    assert s["longest_gap_start"] == pd.Timestamp("2023-01-05 04:00", tz="UTC")
    assert s["zero_volume_bars"] == 3
    assert len(iss[iss.check == "zero_volume"]) == 1
    assert s["ohlc_bad_bars"] >= 1
    assert t in set(iss.loc[iss.check == "outlier", "start"])


def test_consistency_exact_and_mismatch(cfg):
    m1 = make_bars("2023-01-01", 600, "1min")
    h1 = _ohlc_from_1m(m1, "1h")
    iss, s = quality.check_consistency(m1, h1, "1m", "1h", "X", cfg)
    assert s["agg_compared"] == 10 and s["agg_mismatch"] == 0
    h1b = h1.copy()
    h1b.iloc[3, 1] *= 1.001
    iss, s = quality.check_consistency(m1, h1b, "1m", "1h", "X", cfg)
    assert s["agg_mismatch"] == 1 and "high" in iss.iloc[0]["detail"]
    gappy = m1.drop(m1.index[130:140])                              # корзина 02:00 неполная
    iss, s = quality.check_consistency(gappy, h1, "1m", "1h", "X", cfg)
    assert s["agg_compared"] == 9 and s["agg_incomplete"] == 1


def test_run_all_writes_reports(cfg):
    m1 = make_bars("2021-10-15", 2 * 1440, "1min")
    for tf, df in {"1m": m1, "15m": _ohlc_from_1m(m1, "15m"), "1h": _ohlc_from_1m(m1, "1h"),
                   "1d": _ohlc_from_1m(m1, "1d")}.items():
        _write_raw(cfg, "SOL", tf, df)
    summ = quality.run_all(cfg, symbols=["SOL"])
    assert cfg["paths"]["data_quality"].exists() and cfg["paths"]["data_quality_summary"].exists()
    bars = summ[summ.kind == "bars"].set_index("tf")
    assert bars.loc["1h", "n_bars"] == 48 and bars.loc["1d", "n_bars"] == 2
    cons = summ[summ.kind == "consistency"]
    assert (cons["agg_mismatch"] == 0).all() and len(cons) == 3
