"""Кэш сетки, статистики окна, перестановка суток."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import engine
from src.backtest.metrics import metrics
from src.cv import grid as G
from src.cv.permute import permute_days
from src.strategies import s1_zscore
from tests.fakes import make_bars


@pytest.fixture
def setup(cfg):
    cfg["universe"]["warmup_only_until"] = {}
    df = make_bars("2023-01-01", 24 * 60, "1h", seed=50)
    f = pd.Series(1e-4, index=pd.date_range("2023-01-01 08:00", periods=180, freq="8h", tz="UTC"))
    axes = {"w": [24, 48], "z_entry": [1.5, 2.0], "z_exit": [0.0, 0.5]}
    gd = G.build_grid(df, f, strategy="s1_zscore", tf="1h", symbol="X", cfg=cfg,
                      slippage=0.0005, axes=axes)
    return cfg, df, f, gd


def test_expand_grid():
    assert G.expand_grid({}) == [{}]
    assert len(G.expand_grid({"a": [1, 2], "b": [3, 4, 5]})) == 6


def test_grid_matches_engine(setup):
    cfg, df, f, gd = setup
    assert gd.pos.shape == (len(df), 8) and gd.round_trip == pytest.approx(0.003)
    j = 5
    p = gd.params[j]
    bt = engine.run(df, s1_zscore.signal(df, **p), fee_per_side=0.001, slippage_per_side=0.0005,
                    funding=f, tf="1h")
    assert np.allclose(gd.net[:, j], bt["net"])
    assert np.allclose(gd.target[:, j], s1_zscore.target_move(df, **p), equal_nan=True)
    assert tuple(gd.coords[j]) == (1, 0, 1)            # w=48, z_entry=1.5, z_exit=0.5


def test_window_stats_match_metrics(setup):
    cfg, df, f, gd = setup
    a, b = df.index[300], df.index[1000]
    st = G.window_stats(gd, a, b, periods_per_year=8760)
    j = 3
    bt = engine.run(df, s1_zscore.signal(df, **gd.params[j]), fee_per_side=0.001,
                    slippage_per_side=0.0005, funding=f, tf="1h")
    mask = (df.index >= a) & (df.index < b)
    m = metrics(bt, periods_per_year=8760, mask=mask)
    assert st.loc[j, "sharpe"] == pytest.approx(m["sharpe"])
    assert st.loc[j, "trades_per_year"] == pytest.approx(m["trades_per_year"], rel=0.1)
    med = np.nanmedian(gd.target[300:1000, j])
    assert st.loc[j, "cost_ratio"] == pytest.approx(0.003 / med)


def test_permute_days_preserves_marginals():
    df = make_bars("2023-01-01 05:00", 24 * 30 + 10, "1h", seed=51)
    df["is_downtime"] = False
    rng = np.random.default_rng(0)
    p = permute_days(df, rng)
    assert p.index.equals(df.index)
    r0 = np.sort(np.log(df["close"]).diff().dropna().to_numpy())
    r1 = np.sort(np.log(p["close"]).diff().dropna().to_numpy())
    assert np.allclose(r0, r1)                             # те же доходности, другой порядок
    assert not np.allclose(df["close"].to_numpy(), p["close"].to_numpy())
    # неполные края на месте
    assert np.allclose(p.loc[:"2023-01-01 23:00", "close"], df.loc[:"2023-01-01 23:00", "close"])
    assert (p["high"] >= p[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (p["low"] <= p[["open", "close"]].min(axis=1) + 1e-9).all()
