"""S1 Z-score: переходы, параметры, прогрев, простой, look-ahead."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.checks import truncation_check
from src.strategies import s1_zscore as s1
from tests.fakes import make_bars


def _df_from_close(close, down=None):
    idx = pd.date_range("2023-01-01", periods=len(close), freq="1h", tz="UTC")
    df = pd.DataFrame({"close": np.asarray(close, float)}, index=idx)
    if down is not None:
        df["is_downtime"] = np.asarray(down, bool)
    return df


def test_transitions_on_given_z_path(monkeypatch):
    #                0     1     2     3     4     5     6     7     8     9    10
    zpath = [np.nan, -2.5, -1.5, -0.6, -0.4, -1.5, 0.3, 2.5, 1.0, 0.4, -2.6]
    df = _df_from_close(np.arange(len(zpath)))
    monkeypatch.setattr(s1, "zscore", lambda d, w: pd.Series(zpath, index=d.index))
    pos = s1.signal(df, w=5, z_entry=2.0, z_exit=0.5).tolist()
    # t1 вход лонг; t2-3 полоса гистерезиса — держим; t4 z ≥ −0.5 — выход;
    # t5 z = −1.5 из флэта — не вход; t7 шорт; t8 держим; t9 z ≤ 0.5 — выход;
    # t10 лонг
    assert pos == [0, 1, 1, 1, 0, 0, 0, -1, -1, 0, 1]


def test_jump_reverses_same_bar(monkeypatch):
    zpath = [-2.5, -1.0, 2.6, 1.0, -2.7]
    df = _df_from_close(np.arange(5))
    monkeypatch.setattr(s1, "zscore", lambda d, w: pd.Series(zpath, index=d.index))
    assert s1.signal(df, w=5, z_entry=2.0, z_exit=0.5).tolist() == [1, 1, -1, -1, 1]


@pytest.mark.parametrize("kw", [dict(w=10, z_entry=1.0, z_exit=1.0),
                                dict(w=10, z_entry=1.0, z_exit=-0.1),
                                dict(w=1, z_entry=2.0, z_exit=0.5),
                                dict(w=10.5, z_entry=2.0, z_exit=0.5)])
def test_param_validation(kw):
    df = _df_from_close(np.arange(30.0))
    with pytest.raises(ValueError):
        s1.signal(df, **kw)


def test_warmup_and_reference_z():
    df = make_bars("2023-01-01", 500, "1h", seed=3)
    w = 24
    z = s1.zscore(df, w=w)
    ref = (df["close"] - df["close"].rolling(w).mean()) / df["close"].rolling(w).std()
    pd.testing.assert_series_equal(z, ref, check_names=False)
    assert z.iloc[:s1.warmup(w=w)].isna().all() and not np.isnan(z.iloc[s1.warmup(w=w)])
    pos = s1.signal(df, w=w, z_entry=1.5, z_exit=0.3)
    assert (pos.iloc[:s1.warmup(w=w)] == 0).all()
    assert set(pos.unique()) <= {-1.0, 0.0, 1.0} and not pos.isna().any()
    assert pos.index.equals(df.index)


def test_flat_window_std_zero_no_action():
    df = _df_from_close([100.0] * 10 + [90.0])
    assert (s1.signal(df, w=5, z_entry=1.0, z_exit=0.2).iloc[:10] == 0).all()


def _downtime_df():
    """Падение цены, затем простой, затем рабочие бары."""
    close = [100, 101, 99, 100, 102, 101, 100, 99, 101, 100, 90, 90, 90, 90, 100, 101]
    down = [False] * 11 + [True] * 3 + [False] * 2
    return _df_from_close(close, down)


def test_no_entry_on_downtime():
    df = _downtime_df()
    df.loc[df.index[11], "close"] = 50.0               # «сигнал» на баре простоя
    pos = s1.signal(df, w=5, z_entry=1.0, z_exit=0.2)
    assert pos.iloc[11] == pos.iloc[10]


def test_position_held_through_downtime_even_if_exit_condition():
    df = _downtime_df()
    pos = s1.signal(df, w=5, z_entry=1.0, z_exit=0.2)
    assert pos.iloc[10] == 1                            # лонг на падении до 90
    df2 = df.copy()
    df2.loc[df.index[11:14], "close"] = 120.0           # на простое z был бы ≫ 0
    pos2 = s1.signal(df2, w=5, z_entry=1.0, z_exit=0.2)
    assert (pos2.iloc[11:14] == 1).all()                # держим: выход невозможен


def test_z_after_downtime_equals_series_without_downtime():
    df = make_bars("2023-01-01", 400, "1h", seed=8)
    down = np.zeros(400, bool)
    down[150:170] = True
    down[300:302] = True
    df["is_downtime"] = down
    z = s1.zscore(df, w=30)
    ref = s1.zscore(df[~down].drop(columns="is_downtime"), w=30).reindex(df.index)
    pd.testing.assert_series_equal(z, ref)
    assert z[down].isna().all()


def test_truncation_synthetic_with_downtime():
    df = make_bars("2023-01-01", 3000, "1h", seed=21)
    down = np.zeros(3000, bool)
    down[1000:1040] = True
    df["is_downtime"] = down
    truncation_check(s1.signal, df, dict(w=48, z_entry=2.0, z_exit=0.5))


@pytest.mark.data
def test_truncation_real_btc_1h():
    try:
        from src.data.load import load
        df = load("BTCUSDT", "1h")
    except Exception:
        pytest.skip("нет clean BTCUSDT 1h")
    truncation_check(s1.signal, df, dict(w=72, z_entry=2.0, z_exit=0.5))
