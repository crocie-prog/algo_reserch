"""S5 Pivot: формулы, смещение уровней на сутки, полнота дня, переходы, перенос, простой."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.checks import truncation_check
from src.backtest.metrics import session_carry_stats
from src.strategies import s5_pivot as s5
from tests.fakes import make_bars


def _day(h, l, c, start, n=24):
    """Сутки 1h-баров с заданными H, L, C дня (остальные бары внутри)."""
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    mid = (h + l) / 2
    close = np.full(n, mid)
    close[-1] = c
    high, low = np.full(n, mid), np.full(n, mid)
    high[3], low[5] = h, l
    return pd.DataFrame({"open": close, "high": np.maximum(high, close),
                         "low": np.minimum(low, close), "close": close}, index=idx)


def test_levels_formula_and_one_day_shift():
    df = pd.concat([_day(110, 90, 104, "2023-01-01"), _day(120, 100, 101, "2023-01-02")])
    lv = s5.daily_levels(df, tf="1h")
    assert lv.loc["2023-01-01"].isna().all().all()            # первый день — нет D−1
    p = (110 + 90 + 104) / 3
    day2 = lv.loc["2023-01-02"]
    assert np.allclose(day2["P"], p) and np.allclose(day2["S1"], 2 * p - 110)
    assert np.allclose(day2["R1"], 2 * p - 90)
    assert lv.loc[pd.Timestamp("2023-01-02 23:00", tz="UTC"), "P"] == pytest.approx(p)


def test_bar_0000_uses_previous_day():
    df = make_bars("2023-01-01", 72, "1h", seed=2)
    lv = s5.daily_levels(df, tf="1h")
    d2 = df.loc["2023-01-02"]
    p2 = (d2["high"].max() + d2["low"].min() + d2["close"].iloc[-1]) / 3
    assert lv.loc[pd.Timestamp("2023-01-03 00:00", tz="UTC"), "P"] == pytest.approx(p2)
    assert lv.loc[pd.Timestamp("2023-01-02 23:00", tz="UTC"), "P"] != pytest.approx(p2)


def test_incomplete_previous_day_gives_nan():
    df = make_bars("2023-01-01 05:00", 19 + 24 + 24, "1h", seed=3)   # 1-й день неполный
    lv = s5.daily_levels(df, tf="1h")
    assert lv.loc["2023-01-02"].isna().all().all()
    assert lv.loc["2023-01-03"].notna().all().all()


def _path(close, P, S1, R1, start="2023-01-02 00:00"):
    idx = pd.date_range(start, periods=len(close), freq="1h", tz="UTC")
    df = pd.DataFrame({"close": close}, index=idx, dtype=float)
    lv = pd.DataFrame({"P": P, "S1": S1, "R1": R1}, index=idx, dtype=float)
    return df, lv


def test_transitions(monkeypatch):
    close = [100, 94, 97, 100.5, 106, 99, 93]
    df, lv = _path(close, [100] * 7, [95] * 7, [105] * 7)
    monkeypatch.setattr(s5, "daily_levels", lambda d, tf: lv)
    # t1 лонг; t2 держим; t3 ≥ P — выход; t4 > R1 — шорт; t5 ≤ P — выход; t6 < S1 — лонг
    assert s5.signal(df, tf="1h").tolist() == [0, 1, 1, 0, -1, 0, 1]


def test_reversal_same_bar(monkeypatch):
    df, lv = _path([94, 106], [100, 100], [95, 95], [105, 105])
    monkeypatch.setattr(s5, "daily_levels", lambda d, tf: lv)
    assert s5.signal(df, tf="1h").tolist() == [1, -1]


def test_carry_over_midnight_exit_by_new_p(monkeypatch):
    df, lv = _path([94, 94, 94], [100, 100, 93], [95, 95, 90], [105, 105, 96],
                   start="2023-01-02 22:00")
    monkeypatch.setattr(s5, "daily_levels", lambda d, tf: lv)
    pos = s5.signal(df, tf="1h")
    # 22:00 лонг; 23:00 держим; 00:00 новый P = 93 ≤ 94 → выход на первом баре суток
    assert pos.tolist() == [1, 1, 0]
    st = session_carry_stats(df.index, pos, early_bars=4)
    assert st["carried_share"] == 1.0 and st["carried_closed_early_share"] == 1.0


def test_downtime_frozen_and_hl_unaffected():
    df = make_bars("2023-01-01", 72, "1h", seed=5)
    down = np.zeros(72, bool)
    down[30:34] = True
    px = df["close"].iloc[29]
    df.loc[df.index[down], ["open", "high", "low", "close"]] = px
    df["is_downtime"] = down
    lv = s5.daily_levels(df, tf="1h")
    d2 = df.loc["2023-01-02"]
    assert lv.loc["2023-01-03", "P"].iloc[0] == pytest.approx(
        (d2["high"].max() + d2["low"].min() + d2["close"].iloc[-1]) / 3)
    pos = s5.signal(df, tf="1h")
    assert (pos.iloc[30:34] == pos.iloc[29]).all()


def test_daily_tf_uses_previous_bar():
    df = make_bars("2023-01-01", 10, "1D", seed=6)
    lv = s5.daily_levels(df, tf="1d")
    p_prev = (df["high"] + df["low"] + df["close"]).shift(1) / 3
    assert np.allclose(lv["P"], p_prev, equal_nan=True)


def test_config_params(cfg):
    assert s5.config_params("1h", cfg) == {"tf": "1h", "day_mode": "intraday"}
    assert s5.config_params("1d", cfg) == {"tf": "1d", "day_mode": "carry"}
    assert s5.carry_kwargs("1h", cfg, {}) == {"early_bars": 4}


def test_truncation_synthetic_with_downtime():
    df = make_bars("2023-01-01", 3000, "1h", seed=25)
    down = np.zeros(3000, bool)
    down[1500:1510] = True
    df["is_downtime"] = down
    truncation_check(s5.signal, df, {"tf": "1h"})


@pytest.mark.data
def test_real_levels_match_clean_1d_and_truncation():
    try:
        from src.data.load import load
        h1 = load("BTCUSDT", "1h")
        d1 = load("BTCUSDT", "1d")
    except Exception:
        pytest.skip("нет clean BTCUSDT")
    lv = s5.daily_levels(h1, tf="1h")
    p_d = (d1["high"] + d1["low"] + d1["close"]) / 3
    p_d.index = p_d.index + pd.Timedelta(days=1)
    first = lv["P"].groupby(h1.index.floor("1D")).first().dropna()
    common = first.index.intersection(p_d.index)
    assert len(common) > 1000
    assert np.allclose(first.loc[common], p_d.loc[common], rtol=1e-12)
    truncation_check(s5.signal, h1, {"tf": "1h"})


# ── режим (B): внутридневной ────────────────────────────────────────────

def test_intraday_closes_at_2300(monkeypatch):
    df, lv = _path([94, 94, 94], [100, 100, 93], [95, 95, 90], [105, 105, 96],
                   start="2023-01-02 22:00")
    monkeypatch.setattr(s5, "daily_levels", lambda d, tf: lv)
    # 22:00 лонг; 23:00 — принудительное закрытие; 00:00 94 ≥ P=93 — нет входа
    assert s5.signal(df, tf="1h", day_mode="intraday").tolist() == [1, 0, 0]


def test_intraday_no_entry_on_last_bar(monkeypatch):
    df, lv = _path([94, 94], [100, 100], [95, 95], [105, 105], start="2023-01-02 23:00")
    monkeypatch.setattr(s5, "daily_levels", lambda d, tf: lv)
    # 23:00: условие входа есть, но бар последний — позиция не открывается;
    # 00:00: новый день, вход по обычному правилу
    assert s5.signal(df, tf="1h", day_mode="intraday").tolist() == [0, 1]


def test_intraday_never_carries_synthetic():
    df = make_bars("2023-01-01", 24 * 60, "1h", seed=30)
    pos = s5.signal(df, tf="1h", day_mode="intraday")
    last = pos.index.hour == 23
    assert (pos[last] == 0).all()
    st = session_carry_stats(df.index, pos, early_bars=4)
    assert st["n_carried"] == 0
    assert (pos != 0).any()                                # стратегия торгует


def test_intraday_downtime_on_last_bar_closes_next_working_bar(monkeypatch):
    df, lv = _path([94, 94, 94, 99], [100] * 4, [95] * 4, [105] * 4, start="2023-01-02 22:00")
    df["is_downtime"] = [False, True, False, False]       # 23:00 — простой
    monkeypatch.setattr(s5, "daily_levels", lambda d, tf: lv)
    # 22:00 лонг; 23:00 простой — держим; 00:00 закрытие (и сразу вход по 94 < 95);
    # 01:00 99 внутри — держим новую позицию
    assert s5.signal(df, tf="1h", day_mode="intraday").tolist() == [1, 1, 1, 1]


def test_intraday_downtime_pending_close_without_reentry(monkeypatch):
    df, lv = _path([94, 94, 98], [100] * 3, [95] * 3, [105] * 3, start="2023-01-02 22:00")
    df["is_downtime"] = [False, True, False]
    monkeypatch.setattr(s5, "daily_levels", lambda d, tf: lv)
    assert s5.signal(df, tf="1h", day_mode="intraday").tolist() == [1, 1, 0]


@pytest.mark.parametrize("mode", ["frozen_levels", "C", "x"])
def test_day_mode_validation(mode):
    with pytest.raises(ValueError):
        s5.signal(make_bars("2023-01-01", 72, "1h"), tf="1h", day_mode=mode)


def test_truncation_intraday():
    truncation_check(s5.signal, make_bars("2023-01-01", 3000, "1h", seed=26),
                     {"tf": "1h", "day_mode": "intraday"})
