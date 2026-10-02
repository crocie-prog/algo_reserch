"""Walk-forward: причинность отбора, сигнал/шум, размеры, сшивка."""
import numpy as np
import pandas as pd
import pytest

from src.cv import grid as G
from src.cv.walkforward import folds, select_fold, walk_forward

AXES = {"w": [24, 48], "z_entry": [1.5, 2.0], "z_exit": [0.0, 0.5]}
TRAIN_END = pd.Timestamp("2024-01-01", tz="UTC")


def _df(close):
    idx = pd.date_range("2021-01-01", periods=len(close), freq="1h", tz="UTC")
    c = np.asarray(close, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0005,
                         "low": np.minimum(o, c) * 0.9995, "close": c,
                         "volume": 1.0, "turnover": c}, index=idx)


def random_walk(seed, n=24 * 365 * 3, sigma=0.01):
    rng = np.random.default_rng(seed)
    return _df(100 * np.exp(np.cumsum(rng.normal(0, sigma, n))))


def mean_reverting(seed, n=24 * 365 * 3):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(0, 0.01, n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + e[i]
    return _df(100 * np.exp(x))


@pytest.fixture
def cfg0(cfg):
    cfg["universe"]["warmup_only_until"] = {}
    cfg["costs"]["fee_per_side"] = 0.0
    cfg["costs"]["include_funding"] = False
    cfg["selection"]["min_trades_per_year"]["1h"] = 1
    return cfg


def _grid(df, cfg, slippage=0.0):
    return G.build_grid(df, None, strategy="s1_zscore", tf="1h", symbol="X", cfg=cfg,
                        slippage=slippage, axes=AXES)


def _folds(df):
    return folds(df.index, scheme="expanding", step_months=3, min_train_months=12,
                 train_end=TRAIN_END)


def test_selection_ignores_data_after_val_start(cfg0):
    df = random_walk(1)
    fold = [f for f in _folds(df) if f.val_start == pd.Timestamp("2023-01-01", tz="UTC")][0]
    df2 = df.copy()
    after = df2.index >= fold.val_start
    rng = np.random.default_rng(99)
    df2.loc[after, ["open", "high", "low", "close"]] *= np.exp(rng.normal(0, 0.05, (after.sum(), 1)))
    df2["high"] = df2[["open", "high", "close"]].max(axis=1)
    df2["low"] = df2[["open", "low", "close"]].min(axis=1)
    a = select_fold(_grid(df, cfg0), fold, cfg0, periods_per_year=8760)
    b = select_fold(_grid(df2, cfg0), fold, cfg0, periods_per_year=8760)
    assert a.topk == b.topk and a.trade == b.trade and a.n_pass == b.n_pass
    assert a.train_dsr == pytest.approx(b.train_dsr) and a.n_eff == b.n_eff


def test_planted_signal_is_traded_and_profitable(cfg0):
    df = mean_reverting(2)
    res = walk_forward(_grid(df, cfg0), df, None, _folds(df), cfg0, slippage=0.0,
                       periods_per_year=8760)
    ft = res.folds_table
    assert ft["trade"].mean() == 1.0
    assert (ft["train_dsr"] > 0.99).all()
    assert (ft["ens_sharpe"] > 0).all()


def test_pure_noise_no_trade_share_about_half(cfg0):
    shares = []
    for seed in range(10):
        df = random_walk(100 + seed)
        gd = _grid(df, cfg0)
        shares += [not select_fold(gd, f, cfg0, periods_per_year=8760).trade for f in _folds(df)]
    share = float(np.mean(shares))
    assert len(shares) >= 80
    assert 0.3 <= share <= 0.7, share


def test_topk_size_and_cost_filter(cfg0):
    df = random_walk(3)
    gd = _grid(df, cfg0)
    f = _folds(df)[-1]
    s = select_fold(gd, f, cfg0, periods_per_year=8760)
    assert s.n_pass == 8 and s.k == 2                         # min(10, ⌈8/4⌉)
    assert s.topk[0] == s.best and len(s.topk) == 2
    cfg0["selection"]["cost_ratio_max"] = 1e-9                # издержки 0 → ratio 0 → проходит
    assert select_fold(gd, f, cfg0, periods_per_year=8760).n_pass == 8
    gd2 = _grid(df, cfg0, slippage=0.01)                      # дорогие сделки → всё отсечено
    s2 = select_fold(gd2, f, cfg0, periods_per_year=8760)
    assert s2.n_pass == 0 and not s2.trade and s2.n_excl_cost == 8


def test_stitching_zero_outside_validation(cfg0):
    df = random_walk(4)
    fl = _folds(df)
    res = walk_forward(_grid(df, cfg0), df, None, fl, cfg0, slippage=0.0,
                       periods_per_year=8760)
    pos = res.bt_ensemble["pos"]
    assert (pos[pos.index < fl[0].val_start] == 0).all()
    for s in res.selections:
        m = (pos.index >= s.fold.val_start) & (pos.index < s.fold.val_end)
        if not s.trade:
            assert (pos[m] == 0).all()
    assert len(res.folds_table) == len(fl) == 8
    assert set(res.scatter["fold"]) == {str(f.val_start.date()) for f in fl}


def test_stitch_boundary_fee(cfg):
    cfg["universe"]["warmup_only_until"] = {}
    cfg["costs"]["include_funding"] = False
    cfg["selection"]["min_trades_per_year"]["1h"] = 1
    df = mean_reverting(5)
    res = walk_forward(_grid(df, cfg), df, None, _folds(df), cfg, slippage=0.0,
                       periods_per_year=8760)
    bt = res.bt_ensemble
    assert np.allclose(bt["fee"], 0.001 * bt["pos"].diff().abs().fillna(bt["pos"].abs()))


def test_best_on_train_rule(cfg0):
    cfg0["selection"]["train_dsr_min"] = 0.999          # ансамбль отказывает почти всегда
    df = random_walk(11)
    gd = _grid(df, cfg0)
    fl = _folds(df)
    res = walk_forward(gd, df, None, fl, cfg0, slippage=0.0, periods_per_year=8760)
    pos = res.bt_best["pos"]
    for s in res.selections:
        m = (pos.index >= s.fold.val_start) & (pos.index < s.fold.val_end)
        assert s.best == int(np.nanargmax(np.where(s.passed, s.train_sharpe, -np.inf)))
        # без правила отказа: торгуем всегда при N_pass ≥ 1, даже если train-DSR < 0.5
        assert np.array_equal(pos[m].to_numpy(), np.sign(gd.pos[m, s.best]))
    assert res.folds_table["trade_best"].all()
    assert (~res.folds_table["trade"]).all()           # ансамбль отказал, «лучшая» — торгует
    # с требованием DSR правило «лучшая» совпадает по торговле с ансамблем
    r2 = walk_forward(gd, df, None, fl, cfg0, slippage=0.0, periods_per_year=8760,
                      best_requires_dsr=True)
    assert (r2.folds_table["trade_best"] == r2.folds_table["trade"]).all()
    # фильтр торгуемости закрывает и правило «лучшая»
    blk = {f.val_start: False for f in fl}
    r3 = walk_forward(gd, df, None, fl, cfg0, slippage=0.0, periods_per_year=8760, tradable=blk)
    assert (r3.bt_best["pos"] == 0).all() and not r3.folds_table["trade_best"].any()
