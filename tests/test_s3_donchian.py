"""S3 Donchian: канал до t, переходы, стоп-и-переворот, прогрев, простой, look-ahead."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.checks import truncation_check
from src.strategies import s3_donchian as s3
from tests.fakes import make_bars


def _ohlc(close, high=None, low=None, down=None):
    idx = pd.date_range("2023-01-01", periods=len(close), freq="1h", tz="UTC")
    c = np.asarray(close, float)
    df = pd.DataFrame({"open": c, "high": c if high is None else np.asarray(high, float),
                       "low": c if low is None else np.asarray(low, float), "close": c}, index=idx)
    if down is not None:
        df["is_downtime"] = np.asarray(down, bool)
    return df


def test_channel_excludes_bar_t():
    df = make_bars("2023-01-01", 500, "1h", seed=4)
    ch = s3.channel(df, w=20)
    pd.testing.assert_series_equal(ch["hh"], df["high"].shift(1).rolling(20).max(), check_names=False)
    pd.testing.assert_series_equal(ch["ll"], df["low"].shift(1).rolling(20).min(), check_names=False)


def test_longs_occur_on_uptrend():
    close = 100 + np.arange(60.0)                       # монотонный рост
    pos = s3.signal(_ohlc(close), w=5)
    assert (pos.iloc[5:] == 1).all()                    # канал с t включительно дал бы 0


def test_transitions_breakout_hold_reverse_tie():
    #         0   1   2   3   4   5    6    7   8   9   10
    close = [10, 11, 12, 11, 10, 13, 12.5, 11, 9, 9.5, 13.1]
    pos = s3.signal(_ohlc(close), w=3).tolist()
    # канал по барам t−3..t−1:
    # t3: [10, 12] → 11 внутри; t4: [11, 12] → 10 < 11 → шорт;
    # t5: [10, 12] → 13 > 12 → переворот в лонг; t6: [10, 13] → внутри;
    # t7: [10, 13] → внутри; t8: [11, 13] → 9 < 11 → переворот в шорт;
    # t9: [9, 12.5] → внутри; t10: [9, 11] → 13.1 > 11 → лонг
    assert pos == [0, 0, 0, 0, -1, 1, 1, 1, -1, -1, 1]


def test_tie_is_not_breakout():
    close = [10, 12, 11, 12, 12]                        # close = hh — не пробой
    assert s3.signal(_ohlc(close), w=3).tolist() == [0, 0, 0, 0, 0]


def test_stop_and_reverse_never_flat_after_first_entry():
    df = make_bars("2023-01-01", 5000, "1h", seed=9)
    pos = s3.signal(df, w=24).to_numpy()
    first = np.flatnonzero(pos != 0)[0]
    assert (pos[first:] != 0).all()


def test_warmup():
    df = make_bars("2023-01-01", 300, "1h", seed=1)
    w = 24
    ch = s3.channel(df, w=w)
    assert ch.iloc[:w].isna().all().all() and ch.iloc[w].notna().all()
    assert (s3.signal(df, w=w).iloc[:s3.warmup(w=w)] == 0).all()


def test_downtime_channel_and_freeze():
    df = make_bars("2023-01-01", 600, "1h", seed=12)
    down = np.zeros(600, bool)
    down[200:230] = True
    df["is_downtime"] = down
    ch = s3.channel(df, w=24)
    ref = s3.channel(df[~down].drop(columns="is_downtime"), w=24).reindex(df.index)
    pd.testing.assert_frame_equal(ch, ref)
    assert ch[down].isna().all().all()
    pos = s3.signal(df, w=24)
    assert (pos.iloc[200:230] == pos.iloc[199]).all()


def test_no_action_on_downtime_even_with_breakout():
    close = [10, 11, 10, 11, 10, 20, 20, 20]
    down = [False] * 5 + [True, True, False]
    pos = s3.signal(_ohlc(close, down=down), w=3)
    assert pos.iloc[5] == pos.iloc[4] and pos.iloc[6] == pos.iloc[4]
    assert pos.iloc[7] == 1                             # первое решение — первый рабочий бар


@pytest.mark.parametrize("w", [1, 0, 2.5])
def test_param_validation(w):
    with pytest.raises(ValueError):
        s3.signal(make_bars("2023-01-01", 50, "1h"), w=w)


def test_truncation_synthetic_with_downtime():
    df = make_bars("2023-01-01", 3000, "1h", seed=22)
    down = np.zeros(3000, bool)
    down[1500:1520] = True
    df["is_downtime"] = down
    truncation_check(s3.signal, df, dict(w=48))


@pytest.mark.data
def test_truncation_real_btc_1h():
    try:
        from src.data.load import load
        df = load("BTCUSDT", "1h")
    except Exception:
        pytest.skip("нет clean BTCUSDT 1h")
    truncation_check(s3.signal, df, dict(w=72))
