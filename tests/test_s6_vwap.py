"""S6 VWAP: сессионный VWAP, dev, прогрев сессии, граница дня, простой, look-ahead."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.checks import truncation_check
from src.strategies import s6_vwap as s6
from tests.fakes import make_bars

KW = dict(anchor="session", k=2.0, e=0.5, min_session_bars=4)


def _bars(n, start="2023-01-01", freq="1h", seed=0):
    return make_bars(start, n, freq, seed=seed)


def test_session_vwap_cumulative_and_reset():
    df = _bars(72)
    v = s6.vwap(df, anchor="session")
    for day, g in df.groupby(df.index.floor("1D")):
        ref = g["turnover"].cumsum() / g["volume"].cumsum()
        assert np.allclose(v.loc[g.index, "vwap"], ref)
        assert v.loc[g.index[0], "session_bar"] == 1      # сброс в 00:00
        assert np.isnan(v.loc[g.index[0], "dev"])
        ref_dev = (g["close"] - ref).expanding(2).std()
        assert np.allclose(v.loc[g.index, "dev"], ref_dev, equal_nan=True)
    t23 = pd.Timestamp("2023-01-01 23:00", tz="UTC")
    assert v.loc[t23, "session_bar"] == 24                # 23:00 — свой день


def test_session_vwap_15m():
    df = _bars(200, freq="15min")
    v = s6.vwap(df, anchor="session")
    assert v.loc[pd.Timestamp("2023-01-02 00:00", tz="UTC"), "session_bar"] == 1
    assert v.loc[pd.Timestamp("2023-01-01 23:45", tz="UTC"), "session_bar"] == 96


def _with_path(monkeypatch, close, vw, dev, sb, start="2023-01-01"):
    idx = pd.date_range(start, periods=len(close), freq="1h", tz="UTC")
    df = pd.DataFrame({"close": close, "volume": 1.0, "turnover": close}, index=idx, dtype=float)
    path = pd.DataFrame({"vwap": vw, "dev": dev, "session_bar": sb}, index=idx, dtype=float)
    monkeypatch.setattr(s6, "vwap", lambda d, anchor, w=None: path)
    return df


def test_transitions(monkeypatch):
    #        0    1    2    3    4    5    6    7    8
    close = [100, 100, 100, 97, 98, 99.6, 103, 101, 96]
    vw = [100] * 9
    dev = [1] * 9
    sb = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    df = _with_path(monkeypatch, close, vw, dev, sb)
    pos = s6.signal(df, **KW).tolist()
    # t3: 97 < 98 → лонг; t4: |98−100| = 2 > 0.5 → держим; t5: 0.4 ≤ 0.5 → выход;
    # t6: 103 > 102 → шорт; t7: держим; t8: 96 < 98 → переворот в лонг
    assert pos == [0, 0, 0, 1, 1, 0, -1, -1, 1]


def test_entry_gated_exit_allowed_early(monkeypatch):
    #        0    1    2    3    | новая сессия: 4(sb1) 5(sb2) 6(sb3) 7(sb4)
    close = [100, 100, 97, 97, 103, 100.2, 95, 95]
    vw = [100, 100, 100, 100, 100, 100, 100, 100]
    dev = [1, 1, 1, 1, 1, 1, 1, 1]
    sb = [1, 2, 3, 4, 1, 2, 3, 4]
    df = _with_path(monkeypatch, close, vw, dev, sb)
    pos = s6.signal(df, **KW).tolist()
    # t2: sb3 < 4 → входа нет; t3: sb4 → лонг; t4: 103 > 102 — встречный, выход,
    # но вход в шорт закрыт прогревом (sb1) → 0; t5 близко к VWAP — флэт;
    # t6: 95 — sb3, вход закрыт; t7: sb4 → лонг
    assert pos == [0, 0, 0, 1, 0, 0, 0, 1]


def _path_at(monkeypatch, times, close, vw, dev, sb):
    idx = pd.DatetimeIndex(times, tz="UTC")
    df = pd.DataFrame({"close": close, "volume": 1.0, "turnover": close}, index=idx, dtype=float)
    path = pd.DataFrame({"vwap": vw, "dev": dev, "session_bar": sb}, index=idx, dtype=float)
    monkeypatch.setattr(s6, "vwap", lambda d, anchor, w=None: path.loc[d.index])
    return df


def test_first_partial_session_blocks_entries(monkeypatch):
    times = ["2023-01-01 20:00", "2023-01-01 21:00", "2023-01-01 22:00", "2023-01-01 23:00"]
    df = _path_at(monkeypatch, times, [100, 100, 100, 97], [100] * 4, [1] * 4, [1, 2, 3, 4])
    assert s6.signal(df, **KW).tolist() == [0, 0, 0, 0]   # ряд начат не в 00:00


def test_carry_over_midnight_exit_by_new_vwap(monkeypatch):
    times = ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 02:00", "2023-01-01 03:00",
             "2023-01-02 00:00", "2023-01-02 01:00"]
    #       сессия 1 (полная, с 00:00)          | сессия 2: VWAP рядом с ценой
    close = [100, 100, 100, 97, 97, 97]
    vw = [100, 100, 100, 100, 97, 97.2]
    dev = [1, 1, 1, 1, np.nan, 1]
    sb = [1, 2, 3, 4, 1, 2]
    df = _path_at(monkeypatch, times, close, vw, dev, sb)
    # 03:00 (sb4): лонг; 00:00 — dev не определён, держим через полночь;
    # 01:00: |97 − 97.2| = 0.2 ≤ 0.5 → выход по VWAP новой сессии
    assert s6.signal(df, **KW).tolist() == [0, 0, 0, 1, 1, 0]


def test_downtime_excluded_and_frozen():
    df = _bars(48, seed=3)
    down = np.zeros(48, bool)
    down[30:33] = True
    df.loc[df.index[down], ["volume", "turnover"]] = 0.0
    df["is_downtime"] = down
    v = s6.vwap(df, anchor="session")
    ref = s6.vwap(df[~down].drop(columns="is_downtime"), anchor="session").reindex(df.index)
    pd.testing.assert_frame_equal(v, ref)
    # бар 33 — 10-й бар второй сессии (с 24), из них 3 простоя → 7-й рабочий
    assert v.loc[df.index[33], "session_bar"] == 7
    pos = s6.signal(df, **KW)
    assert (pos.iloc[30:33] == pos.iloc[29]).all()


def test_zero_volume_session_no_action():
    df = _bars(30, seed=4)
    df.loc[df.index[24:27], ["volume", "turnover"]] = 0.0   # начало сессии без объёма
    v = s6.vwap(df, anchor="session")
    assert v.loc[df.index[24:27], "vwap"].isna().all()


def test_rolling_reference():
    df = _bars(300, seed=5)
    v = s6.vwap(df, anchor="rolling", w=20)
    ref = df["turnover"].rolling(20).sum() / df["volume"].rolling(20).sum()
    assert np.allclose(v["vwap"], ref, equal_nan=True)
    ref_dev = (df["close"] - ref).rolling(20).std()
    assert np.allclose(v["dev"], ref_dev, equal_nan=True)
    pos = s6.signal(df, anchor="rolling", k=1.5, e=0.3, w=20)
    assert (pos.iloc[:s6.warmup(anchor="rolling", w=20)] == 0).all()


@pytest.mark.parametrize("kw", [dict(anchor="session", k=1.0, e=1.0, min_session_bars=4),
                                dict(anchor="session", k=1.0, e=0.5),
                                dict(anchor="rolling", k=1.0, e=0.5),
                                dict(anchor="weekly", k=1.0, e=0.5, w=5)])
def test_param_validation(kw):
    with pytest.raises(ValueError):
        s6.signal(_bars(50), **kw)


def test_config_params(cfg):
    assert s6.config_params("1h", cfg) == {"anchor": "session", "min_session_bars": 4}
    assert s6.config_params("15m", cfg) == {"anchor": "session", "min_session_bars": 16}
    assert s6.config_params("1d", cfg) == {"anchor": "rolling"}


def test_session_carry_stats():
    idx = pd.date_range("2023-01-01 20:00", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": 1.0, "volume": 1.0, "turnover": 1.0}, index=idx)
    # сделка 1: 21:00–02:00 (перенесена, закрыта на 03:00 — 4-й бар сессии);
    # сделка 2: 04:00–05:00 (не перенесена, открыта на конце)
    pos = pd.Series([0, 1, 1, 1, 1, 1, 1, 0, -1, -1], index=idx, dtype=float)
    st = s6.session_carry_stats(df, pos, min_session_bars=4)
    assert st["carried_share"] == pytest.approx(0.5) and st["n_carried"] == 1
    assert st["carried_closed_early_share"] == pytest.approx(1.0)


def test_truncation_synthetic_with_downtime():
    df = _bars(3000, seed=24)
    down = np.zeros(3000, bool)
    down[1500:1510] = True
    df.loc[df.index[down], ["volume", "turnover"]] = 0.0
    df["is_downtime"] = down
    truncation_check(s6.signal, df, KW)


@pytest.mark.data
def test_truncation_real_btc_1h():
    try:
        from src.data.load import load
        df = load("BTCUSDT", "1h")
    except Exception:
        pytest.skip("нет clean BTCUSDT 1h")
    truncation_check(s6.signal, df, KW)
