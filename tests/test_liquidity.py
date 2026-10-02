"""Фильтр торгуемости H2: доля минут без сделок за предыдущий квартал."""
import numpy as np
import pandas as pd

from src.cv.walkforward import walk_forward
from src.data import store
from src.data.liquidity import no_trade_share, tradable
from tests.fakes import make_bars
from tests.test_walkforward import _folds, _grid, random_walk


def test_no_trade_share_and_rule(cfg):
    m = make_bars("2022-01-01", 60 * 24 * 182, "1min", seed=1)        # ~2 квартала
    q1 = (m.index >= "2022-02-01") & (m.index < "2022-02-15")          # 14 дней без сделок в Q1
    m.loc[q1, "volume"] = 0.0
    store.write(cfg["paths"]["clean"], "X", "1m", m)
    sh = no_trade_share(cfg, "X", months=3)
    q1s, q2s = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-04-01", tz="UTC")
    assert sh[q1s] == pytest_approx(14 / 90)
    assert sh[q2s] == 0.0
    # Q2 зависит от Q1 (15.6% > 10%) — нельзя; Q3 от Q2 (0%) — можно
    assert not tradable(sh, q2s, months=3, max_share=0.10)
    assert tradable(sh, pd.Timestamp("2022-07-01", tz="UTC"), months=3, max_share=0.10)
    # нет данных за предыдущий квартал — нельзя
    assert not tradable(sh, q1s, months=3, max_share=0.10)


def pytest_approx(x):
    import pytest
    return pytest.approx(x, rel=1e-9)


def test_walk_forward_blocks_quarter(cfg):
    cfg["universe"]["warmup_only_until"] = {}
    cfg["costs"]["fee_per_side"] = 0.0
    cfg["costs"]["include_funding"] = False
    cfg["selection"]["min_trades_per_year"]["1h"] = 1
    df = random_walk(8)
    gd = _grid(df, cfg)
    fl = _folds(df)
    block = {f.val_start: (i % 2 == 0) for i, f in enumerate(fl)}
    res = walk_forward(gd, df, None, fl, cfg, slippage=0.0, periods_per_year=8760,
                       tradable=block)
    pos = res.bt_ensemble["pos"]
    for i, f in enumerate(fl):
        m = (pos.index >= f.val_start) & (pos.index < f.val_end)
        if not block[f.val_start]:
            assert (pos[m] == 0).all()
    assert res.folds_table["liquidity_ok"].tolist() == [block[f.val_start] for f in fl]
    base = walk_forward(gd, df, None, fl, cfg, slippage=0.0, periods_per_year=8760)
    assert base.folds_table["liquidity_ok"].all()
    assert (base.folds_table["trade"] == res.folds_table["trade"]).all()   # отбор не меняется
