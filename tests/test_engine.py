"""Движок: выравнивание, издержки, обнуление прогрева, проскальзывание, входное состояние."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import engine
from tests.fakes import make_bars


@pytest.fixture
def df():
    return make_bars("2023-01-01", 200, "1h", seed=5)


def test_held_is_lagged_pos(df):
    pos = pd.Series(np.sign(np.sin(np.arange(200))), index=df.index)
    bt = engine.run(df, pos, fee_per_side=0.0)
    assert bt["held"].iloc[0] == 0 and (bt["held"].iloc[1:].values == pos.iloc[:-1].values).all()
    ret = df["close"].pct_change().fillna(0)
    assert np.allclose(bt["gross"], bt["held"] * ret)


def test_validation(df):
    with pytest.raises(ValueError):
        engine.run(df, pd.Series(1.5, index=df.index), fee_per_side=0)
    with pytest.raises(ValueError):
        engine.run(df, pd.Series(np.nan, index=df.index), fee_per_side=0)
    with pytest.raises(ValueError):
        engine.run(df, pd.Series(1.0, index=df.index[:-1]), fee_per_side=0)


def test_warmup_position_not_carried_without_entry_fee(df):
    pos = pd.Series(1.0, index=df.index)                 # лонг открыт уже в прогреве
    af = df.index[100]
    bt = engine.run(df, pos, fee_per_side=0.001, active_from=af)
    assert (bt.loc[:af, "held"] == 0).all()              # на первом рабочем баре ещё флэт
    assert (bt.loc[bt.index < af, "pos"] == 0).all()
    assert bt.loc[af, "fee"] == pytest.approx(0.001)     # вход на close первого рабочего бара
    assert bt.loc[bt.index < af, ["gross", "fee", "net"]].abs().sum().sum() == 0
    assert bt["fee"].sum() == pytest.approx(0.001)


def test_window_starting_in_warmup_ignores_initial_pos(df):
    pos = pd.Series(1.0, index=df.index)
    bt = engine.run(df, pos, fee_per_side=0.001, initial_pos=1.0, active_from=df.index[10])
    assert bt["held"].iloc[0] == 0


def test_slippage_equals_fee_plus_x(df):
    rng = np.random.default_rng(0)
    pos = pd.Series(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], 200), index=df.index)
    a = engine.run(df, pos, fee_per_side=0.001, slippage_per_side=0.0007)
    b = engine.run(df, pos, fee_per_side=0.0017)
    assert np.allclose(a["net"], b["net"], rtol=0, atol=1e-15)
    assert np.allclose(a["equity"], b["equity"], rtol=1e-12)


def test_input_state_matches_full_run(df):
    rng = np.random.default_rng(1)
    pos = pd.Series(rng.choice([-1.0, 0.0, 1.0], 200), index=df.index)
    f = pd.Series(1e-4, index=pd.date_range("2023-01-01 08:00", periods=25, freq="8h", tz="UTC"))
    full = engine.run(df, pos, fee_per_side=0.001, funding=f, tf="1h")
    k = 120
    part = engine.run(df.iloc[k:], pos.iloc[k:], fee_per_side=0.001, funding=f, tf="1h",
                      initial_pos=pos.iloc[k - 1], prev_close=df["close"].iloc[k - 1])
    cols = ["held", "ret", "gross", "turnover", "fee", "funding", "net"]
    assert np.allclose(full.iloc[k:][cols], part[cols], rtol=0, atol=1e-15)
