"""CSCV/PBO и CPCV на синтетике."""
import itertools

import numpy as np
import pandas as pd
import pytest

from src.cv import cpcv as C


def test_cscv_combos_count_and_symmetry():
    m = C.cscv_combos(16)
    assert m.shape == (12870, 16) and (m.sum(axis=1) == 8).all()
    s = {tuple(r) for r in m}
    assert all(tuple(~r) in s for r in m)


def test_block_sums_h0_equal_direct():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (1600, 5))
    res = C.cscv(x, n_groups=8, h=0, periods_per_year=1)
    combos = C.cscv_combos(8)
    bounds = C.group_bounds(1600, 8)
    i = 17
    is_rows = np.concatenate([np.arange(a, b) for g, (a, b) in enumerate(bounds) if combos[i, g]])
    direct = x[is_rows].mean(axis=0) / x[is_rows].std(axis=0, ddof=1)
    blk = C._block_sums(x, bounds, 0)
    left_ok, right_ok = C._is_masks(combos)
    is_s = (np.einsum("cg,gjk->cjk", combos.astype(float), blk["core"]))
    assert np.allclose(C._sharpe_from_sums(is_s)[i], direct)
    assert res["embargo_share"] == 0.0


def test_embargo_removes_h_bars_at_is_oos_boundaries():
    x = np.random.default_rng(1).normal(size=(800, 3))
    res = C.cscv(x, n_groups=8, h=10, periods_per_year=1)
    combos = C.cscv_combos(8)
    # число удалённых баров = 10 × (число IS-групп, граничащих с OOS слева) + 10 × (справа)
    left_ok, right_ok = C._is_masks(combos)
    removed = 10 * ((combos & ~left_ok).sum(axis=1) + (combos & ~right_ok).sum(axis=1))
    assert res["embargo_share"] == pytest.approx(np.mean(removed / 400))


def test_pbo_noise_about_half_and_edge_near_zero():
    pbos = []
    for seed in range(6):
        x = np.random.default_rng(10 + seed).normal(0, 0.01, (4000, 20))
        pbos.append(C.cscv(x, n_groups=16, h=0, periods_per_year=8760)["pbo"])
    assert 0.35 <= np.mean(pbos) <= 0.65
    x = np.random.default_rng(3).normal(0, 0.01, (4000, 20))
    x[:, 7] += 0.004                                       # одна конфигурация с настоящим преимуществом
    r = C.cscv(x, n_groups=16, h=0, periods_per_year=8760)
    assert r["pbo"] < 0.05 and r["p_oos_loss"] < 0.05
    assert r["best_counts"].argmax() == 7


def test_cpcv_paths_cover_each_group_once():
    splits, paths = C.cpcv_splits(6, 2)
    assert len(splits) == 15 and paths.shape == (5, 6)
    for p in range(5):
        for g in range(6):
            assert g in splits[paths[p, g]]
    for g in range(6):
        assert len(set(paths[:, g])) == 5


def test_embargo_bars():
    pos = np.zeros((20, 2))
    pos[2:6, 0] = 1                                         # сделка 4 бара
    pos[1:3, 1] = 1
    pos[5:7, 1] = -1                                        # две сделки по 2 бара
    assert C.embargo_bars(pos) == 4


def test_cpcv_end_to_end_synthetic(cfg):
    from src.cv.grid import build_grid
    from tests.test_walkforward import AXES, mean_reverting
    cfg["universe"]["warmup_only_until"] = {}
    cfg["costs"]["include_funding"] = False
    cfg["selection"]["min_trades_per_year"]["1h"] = 1
    df = mean_reverting(6, n=24 * 365)
    gd = build_grid(df, None, strategy="s1_zscore", tf="1h", symbol="X", cfg=cfg,
                    slippage=0.0, axes=AXES)
    out = C.cpcv(gd, df, None, cfg, start=df.index[500], end=df.index[-1], n_groups=6,
                 k_test=2, h=C.embargo_bars(gd.pos[500:]), periods_per_year=8760)
    assert len(out) == 10 and set(out["rule"]) == {"procedure", "best_is"}
    assert (out["sharpe"] > 0).all()                        # внедрённый сигнал виден на всех путях
