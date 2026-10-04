"""Сквозной прогон H3 на синтетике: H1 → H2 → H3, N сквозное, уровень 2 на 4h."""
import numpy as np
import pandas as pd

from src.cv import h2, h3, stage5
from src.data import quality, store
from src.stats import trials as T
from tests.fakes import make_bars
from tests.test_stage5_e2e import cfg5  # noqa: F401


def test_to_tf_sums_bins():
    idx = pd.date_range("2022-01-01 02:00", periods=6, freq="1h", tz="UTC")
    x = pd.Series([1.0, 2, 3, 4, 5, 6], index=idx)
    y = h3.to_tf(x, "4h")
    assert list(y.index.hour) == [0, 4] and y.tolist() == [3.0, 18.0]


def test_h3_end_to_end(cfg5):  # noqa: F811
    cfg = cfg5
    stage5.run_batch(cfg, tfs=["1h"], slippages=[0.0], schemes=["expanding", "rolling"])
    news = ["NEW1", "NEW2"]
    for i, s in enumerate(news):
        df = make_bars("2021-01-01", 24 * 730, "1h", seed=300 + i)
        df["is_downtime"] = False
        store.write(cfg["paths"]["clean"], s, "1h", df)
    cfg["h2"]["symbols"] = news
    cfg["grids"]["s4_supertrend"]["1h"] = {"n": [24], "m": [3.0, 4.0]}
    allow = lambda c, s, fl: {f.val_start: True for f in fl}
    d1h = lambda s: (store.read(cfg["paths"]["clean"], s, "1h"), None)
    h2.run(cfg, symbols=news, n_perm=0, tradable_fn=allow, data_fn=d1h)
    n_before = int((T.read_trials(cfg["paths"]["trials_log"]).kind == "attempt").sum())

    cfg["grids"]["s3_donchian"]["4h"] = {"w": [6, 12]}
    cfg["grids"]["s4_supertrend"]["4h"] = {"n": [6], "m": [2.0, 3.0]}
    cfg["selection"]["min_trades_per_year"]["4h"] = 1
    syms = h3.symbols_h3(cfg)
    assert syms == ["AAA", "BBB", "CCC", "NEW1", "NEW2"]

    def d4h(s):
        a = quality.aggregate(store.read(cfg["paths"]["clean"], s, "1h"), "4h")
        a = a[a["n"] == 4].drop(columns="n")
        a["is_downtime"] = False
        return a, None

    blocked = pd.Timestamp("2022-07-01", tz="UTC")
    trad = lambda c, s, fl: {f.val_start: not (s == "NEW1" and f.val_start == blocked) for f in fl}
    path = h3.run(cfg, symbols=syms, n_perm=2, tradable_fn=trad, data_fn=d4h)

    log = T.read_trials(cfg["paths"]["trials_log"])
    att = log[log.kind == "attempt"]
    assert len(att) == n_before + 2
    assert set(att.iloc[-2:]["strategy"]) == {"s3_donchian", "s4_supertrend"}
    h3rows = log[log.variant.str.startswith("h3_")]
    assert {"train", "attempt", "sensitivity", "diagnostic"} <= set(h3rows.kind)
    assert (h3rows[h3rows.kind == "diagnostic"].params.str.contains("plateau")).all()
    out = cfg["paths"]["h3_out"]
    ft = pd.read_csv(out / "h3_4h_expanding_slip0" / "s3_donchian" / "folds.csv")
    r = ft[(ft.symbol == "NEW1") & ft.val_start.str.startswith("2022-07-01")]
    assert not r.liquidity_ok.iloc[0]
    pr = pd.read_csv(out / "h3_4h_expanding_slip0" / "s3_donchian" / "pairs.csv")
    assert set(pr.group) == {"H1", "H2"}
    assert np.isclose(pr.set_index("symbol").loc["NEW1", "share_blocked_liq"],
                      1 / (ft.symbol == "NEW1").sum())
    assert len(pd.read_csv(out / "noise.csv")) == 2 * 2
    text = path.read_text(encoding="utf-8")
    assert f"N = {n_before + 2}" in text and f"в журнале attempt: {n_before + 2}" in text
    assert "## s3_donchian" in text and "## s4_supertrend" in text
    assert "| BTC/ETH/SOL |" in text and "| 2 пар H2 |" in text and "Корреляция EW-OOS" in text
