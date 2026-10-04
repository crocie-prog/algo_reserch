"""clean 15m/1h/1d — точная агрегация clean 1m; флаг простоя."""
import numpy as np
import pandas as pd
import pytest

from src.config import tf_delta
from src.data import clean, quality, store
from tests.fakes import make_bars

COLS = ["open", "high", "low", "close", "volume", "turnover"]


@pytest.fixture
def built(cfg):
    """Два года 1m через стык; простой; родной 1h с «чужим» turnover."""
    m1 = pd.concat([make_bars("2021-12-30 00:00", 3 * 1440, "1min", seed=1)])
    # простой: целый час 2021-12-31 05:00 и половина часа 06:00 — плоские бары, объём 0
    for start, n in (("2021-12-31 05:00", 60), ("2021-12-31 06:00", 30)):
        sl = m1.index[(m1.index >= pd.Timestamp(start, tz="UTC"))][:n]
        px = m1.loc[sl[0], "open"]
        m1.loc[sl, ["open", "high", "low", "close"]] = px
        m1.loc[sl, ["volume", "turnover"]] = 0.0
    # минута без сделок, но не плоская (объём 0) — не простой
    t = pd.Timestamp("2021-12-31 07:10", tz="UTC")
    m1.loc[t, ["volume", "turnover"]] = 0.0
    store.write(cfg["paths"]["raw"], "X", "1m", m1)
    native = quality.aggregate(m1, "1h").drop(columns="n")
    native["turnover"] *= 1.01                      # родной расходится
    store.write(cfg["paths"]["raw"], "X", "1h", native)
    for tf in ("1m", "15m", "1h", "4h", "1d"):
        clean.build_clean(cfg, "X", tf)
    return cfg, m1


@pytest.mark.parametrize("tf", ["15m", "1h", "4h", "1d"])
def test_clean_equals_aggregate_of_clean_1m(built, tf):
    cfg, _ = built
    c1m = store.read(cfg["paths"]["clean"], "X", "1m")
    agg = quality.aggregate(c1m, tf)
    agg = agg[agg["n"] == tf_delta(tf) // tf_delta("1m")]
    got = store.read(cfg["paths"]["clean"], "X", tf)
    assert got.index.equals(agg.index)
    for c in COLS:                                   # точное равенство
        assert np.array_equal(got[c].to_numpy(), agg[c].to_numpy()), c


def test_raw_native_untouched_and_not_used(built):
    cfg, _ = built
    raw = store.read(cfg["paths"]["raw"], "X", "1h")
    cl = store.read(cfg["paths"]["clean"], "X", "1h")
    assert not np.array_equal(raw["turnover"].to_numpy(), cl["turnover"].to_numpy())


def test_downtime_flags(built):
    cfg, _ = built
    c1m = store.read(cfg["paths"]["clean"], "X", "1m")
    assert c1m["is_downtime"].sum() == 90
    assert not c1m.loc[pd.Timestamp("2021-12-31 07:10", tz="UTC"), "is_downtime"]
    h = store.read(cfg["paths"]["clean"], "X", "1h")
    t5, t6 = pd.Timestamp("2021-12-31 05:00", tz="UTC"), pd.Timestamp("2021-12-31 06:00", tz="UTC")
    assert h.loc[t5, "is_downtime"] and h.loc[t5, "downtime_share"] == 1.0
    assert not h.loc[t6, "is_downtime"] and h.loc[t6, "downtime_share"] == 0.5
    assert len(h) == 72                               # бары простоя не удалены


def test_report_has_native_diagnostics(built):
    cfg, _ = built
    summ = quality.run_all(cfg, symbols=["X"])
    row = summ[summ.tf == "1m->1h родной"].iloc[0]
    # turnover часа простоя = 0 и после ×1.01 совпадает
    assert row["agg_compared"] == 72 and row["agg_mismatch"] == 71
    bars = summ[(summ.kind == "bars") & (summ.tf == "1h")].iloc[0]
    assert bars["downtime_bars"] == 1


def test_4h_bins_anchor_and_1d_from_4h(built):
    """4h: корзины 00, 04, …, 20 UTC; 1d = точный агрегат 4h (OHLC, суммы, простой)."""
    cfg, _ = built
    h4 = store.read(cfg["paths"]["clean"], "X", "4h")
    d1 = store.read(cfg["paths"]["clean"], "X", "1d")
    assert set(h4.index.hour) == {0, 4, 8, 12, 16, 20} and (h4.index.minute == 0).all()
    assert pd.Timestamp("2022-01-01 00:00", tz="UTC") in h4.index      # стык партиций 1m
    g = h4.groupby(h4.index.floor("1D"))
    agg = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(),
                        "volume": g["volume"].sum(), "turnover": g["turnover"].sum(),
                        "downtime_share": g["downtime_share"].mean(), "n": g.size()})
    agg = agg[agg["n"] == 6]
    assert d1.index.equals(agg.index)
    for c in ["open", "high", "low", "close"]:
        assert np.array_equal(d1[c].to_numpy(), agg[c].to_numpy()), c
    for c in ["volume", "turnover", "downtime_share"]:          # суммы — до округления
        assert np.allclose(d1[c].to_numpy(), agg[c].to_numpy(), rtol=1e-12, atol=0), c
    t = pd.Timestamp("2021-12-31 04:00", tz="UTC")              # простой 05:00–06:30 → 90 из 240
    assert h4.loc[t, "downtime_share"] == pytest.approx(90 / 240) and not h4.loc[t, "is_downtime"]
