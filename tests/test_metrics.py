"""Метрики: ручной пример, деление издержек на перевороте, окно = вырезанный ряд."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import engine
from src.backtest.metrics import max_drawdown, metrics, trades
from tests.fakes import make_bars


def _df(closes):
    idx = pd.date_range("2023-01-01", periods=len(closes), freq="1h", tz="UTC")
    c = pd.Series(closes, index=idx, dtype="float64")
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0,
                         "turnover": c})


def test_hand_example():
    df = _df([100, 110, 88, 88, 96.8])
    pos = pd.Series([1, 1, 0, -1, -1], index=df.index, dtype="float64")
    bt = engine.run(df, pos, fee_per_side=0.0)
    # held: 0,1,1,0,-1; ret: 0, .1, -.2, 0, .1 → net: 0, .1, -.2, 0, -.1
    assert np.allclose(bt["net"], [0, 0.1, -0.2, 0, -0.1])
    m = metrics(bt, periods_per_year=8760)
    eq_end = 1.1 * 0.8 * 0.9
    assert m["ann_return"] == pytest.approx(eq_end ** (8760 / 5) - 1)
    assert m["mdd"] == pytest.approx(eq_end / 1.1 - 1)
    assert m["exposure"] == pytest.approx(3 / 5)
    tr = trades(bt)
    assert list(tr["side"]) == [1, -1] and list(tr["bars"]) == [2, 1]
    assert tr["pnl"].tolist() == pytest.approx([-0.1, -0.1])
    assert m["n_trades"] == 2 and m["win_rate"] == 0.0 and m["open_at_end"]
    assert m["trades_per_year"] == pytest.approx(2 / (5 / 8760))


def test_reversal_cost_split():
    df = _df([100, 100, 100, 100])
    pos = pd.Series([1, -1, -1, 0], index=df.index, dtype="float64")
    bt = engine.run(df, pos, fee_per_side=0.001)
    tr = trades(bt)
    # лонг: вход 0.001 + половина переворота 0.001; шорт: половина 0.001 + выход 0.001
    assert tr["pnl"].tolist() == pytest.approx([-0.002, -0.002])
    assert tr["pnl"].sum() == pytest.approx(-bt["fee"].sum())


def test_resize_same_sign_belongs_to_trade():
    df = _df([100, 100, 100, 100])
    pos = pd.Series([0.5, 1.0, 1.0, 0], index=df.index, dtype="float64")
    bt = engine.run(df, pos, fee_per_side=0.001)
    tr = trades(bt)
    assert len(tr) == 1 and tr["pnl"].iloc[0] == pytest.approx(-bt["fee"].sum())


def test_max_drawdown_includes_start():
    assert max_drawdown(np.array([0.9, 1.2, 0.6])) == pytest.approx(0.6 / 1.2 - 1)
    assert max_drawdown(np.array([0.8])) == pytest.approx(-0.2)


def test_mask_equals_sliced_series_with_same_input_state():
    df = make_bars("2023-01-01", 3000, "1h", seed=7)
    rng = np.random.default_rng(3)
    raw = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], 3000)
    pos = pd.Series(np.repeat(raw[::10], 10)[:3000], index=df.index)
    f = pd.Series(rng.normal(1e-4, 2e-4, 125),
                  index=pd.date_range("2023-01-01 08:00", periods=125, freq="8h", tz="UTC"))
    full = engine.run(df, pos, fee_per_side=0.001, slippage_per_side=0.0002, funding=f, tf="1h")
    k0, k1 = 1234, 2345                                  # окно начинается внутри сделки
    mask = np.zeros(3000, bool)
    mask[k0:k1] = True
    m_mask = metrics(full, periods_per_year=8760, mask=mask)
    part = engine.run(df.iloc[k0:k1], pos.iloc[k0:k1], fee_per_side=0.001,
                      slippage_per_side=0.0002, funding=f, tf="1h",
                      initial_pos=pos.iloc[k0 - 1], prev_close=df["close"].iloc[k0 - 1])
    m_cut = metrics(part, periods_per_year=8760)
    assert m_mask.keys() == m_cut.keys()
    for k in m_mask:
        assert m_mask[k] == pytest.approx(m_cut[k], rel=1e-12, abs=1e-15), k


def test_noncontiguous_mask_rejected():
    df = _df([1, 2, 3, 4])
    bt = engine.run(df, pd.Series(0.0, index=df.index), fee_per_side=0)
    with pytest.raises(ValueError):
        metrics(bt, periods_per_year=8760, mask=np.array([1, 0, 1, 0], bool))
