"""S4 Supertrend: TR/ATR Уайлдера, рекурсия, вход по развороту, простой, look-ahead."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.checks import truncation_check
from src.strategies import s4_supertrend as s4
from src.strategies._common import atr_wilder, true_range
from tests.fakes import make_bars


def _ohlc(h, l, c, down=None):
    idx = pd.date_range("2023-01-01", periods=len(c), freq="1h", tz="UTC")
    df = pd.DataFrame({"open": c, "high": h, "low": l, "close": c}, index=idx, dtype=float)
    if down is not None:
        df["is_downtime"] = np.asarray(down, bool)
    return df


def test_true_range_and_wilder_hand():
    df = _ohlc([10, 12, 11, 15], [8, 9, 7, 12], [9, 11, 8, 14])
    assert true_range(df).tolist() == [2, 3, 4, 7]        # H−L; max(3,3,0); max(4,0,4); max(3,7,4)
    a = atr_wilder(df, 2)
    assert np.isnan(a.iloc[0]) and a.iloc[1] == pytest.approx(2.5)
    assert a.iloc[2] == pytest.approx((2.5 + 4) / 2) and a.iloc[3] == pytest.approx((3.25 + 7) / 2)


def _naive_supertrend(df, n, m):
    """Независимая реализация (по формулам docstring) для сверки."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[np.nan, c[:-1]]),
                                      np.abs(l - np.r_[np.nan, c[:-1]])))
    tr[0] = h[0] - l[0]
    atr = np.full(len(c), np.nan)
    atr[n - 1] = tr[:n].mean()
    for t in range(n, len(c)):
        atr[t] = (atr[t - 1] * (n - 1) + tr[t]) / n
    hl2 = (h + l) / 2
    up, dn, d = [np.full(len(c), np.nan) for _ in range(3)]
    s = n - 1
    up[s], dn[s] = hl2[s] + m * atr[s], hl2[s] - m * atr[s]
    d[s] = 1 if c[s] >= hl2[s] else -1
    for t in range(s + 1, len(c)):
        bu, bd = hl2[t] + m * atr[t], hl2[t] - m * atr[t]
        up[t] = bu if bu < up[t - 1] or c[t - 1] > up[t - 1] else up[t - 1]
        dn[t] = bd if bd > dn[t - 1] or c[t - 1] < dn[t - 1] else dn[t - 1]
        d[t] = (-1 if c[t] < dn[t] else 1) if d[t - 1] > 0 else (1 if c[t] > up[t] else -1)
    return d, up, dn


def test_supertrend_matches_naive_and_ratchet():
    df = make_bars("2023-01-01", 3000, "1h", seed=14)
    st = s4.supertrend(df, n=14, m=2.5)
    d, up, dn = _naive_supertrend(df, 14, 2.5)
    assert np.allclose(st["dir"], d, equal_nan=True)
    assert np.allclose(st["up"], up, equal_nan=True) and np.allclose(st["dn"], dn, equal_nan=True)
    dd, u, l = st["dir"].values, st["up"].values, st["dn"].values
    same_up = (dd[1:] > 0) & (dd[:-1] > 0)
    assert (l[1:][same_up] >= l[:-1][same_up] - 1e-12).all()   # dn не убывает в аптренде
    same_dn = (dd[1:] < 0) & (dd[:-1] < 0)
    assert (u[1:][same_dn] <= u[:-1][same_dn] + 1e-12).all()   # up не растёт в даунтренде


def test_entry_on_flip_after_warmup_then_stop_and_reverse():
    df = make_bars("2023-01-01", 3000, "1h", seed=15)
    n, m = 10, 2.0
    pos = s4.signal(df, n=n, m=m).to_numpy()
    d = s4.supertrend(df, n=n, m=m)["dir"].to_numpy()
    wu = s4.warmup(n=n)
    assert (pos[:wu] == 0).all()
    flips = np.flatnonzero((d[1:] != d[:-1]) & ~np.isnan(d[:-1])) + 1
    first = flips[flips >= wu][0]
    assert (pos[wu:first] == 0).all()                    # посреди тренда не входим
    assert pos[first] == d[first]
    assert (pos[first:] == d[first:]).all()              # дальше позиция = dir


def test_downtime_working_series_and_gap_in_tr():
    df = make_bars("2023-01-01", 800, "1h", seed=16)
    down = np.zeros(800, bool)
    down[300:320] = True
    df["is_downtime"] = down
    st = s4.supertrend(df, n=20, m=3.0)
    ref = s4.supertrend(df[~down].drop(columns="is_downtime"), n=20, m=3.0).reindex(df.index)
    pd.testing.assert_frame_equal(st, ref)
    tr = true_range(df)
    pc = df["close"].iloc[299]
    h, l = df["high"].iloc[320], df["low"].iloc[320]
    assert tr.iloc[320] == pytest.approx(max(h - l, abs(h - pc), abs(l - pc)))
    pos = s4.signal(df, n=20, m=3.0)
    assert (pos.iloc[300:320] == pos.iloc[299]).all()


def test_start_sensitivity_diagnostic():
    df = make_bars("2023-01-01", 6000, "1h", seed=17)
    n, k = 20, 1000
    full = s4.signal(df, n=n, m=3.0)
    late = s4.signal(df.iloc[k:], n=n, m=3.0)
    tail = slice(k + 10 * n, None)
    agree = (full.iloc[tail].to_numpy() == late.reindex(df.index).iloc[tail].to_numpy()).mean()
    assert agree >= 0.99


@pytest.mark.parametrize("kw", [dict(n=1, m=3.0), dict(n=10, m=0.0), dict(n=2.5, m=3.0)])
def test_param_validation(kw):
    with pytest.raises(ValueError):
        s4.signal(make_bars("2023-01-01", 100, "1h"), **kw)


def test_truncation_synthetic_with_downtime():
    df = make_bars("2023-01-01", 3000, "1h", seed=23)
    down = np.zeros(3000, bool)
    down[1200:1230] = True
    df["is_downtime"] = down
    truncation_check(s4.signal, df, dict(n=24, m=3.0))


@pytest.mark.data
def test_truncation_real_btc_1h():
    try:
        from src.data.load import load
        df = load("BTCUSDT", "1h")
    except Exception:
        pytest.skip("нет clean BTCUSDT 1h")
    truncation_check(s4.signal, df, dict(n=72, m=3.0))
