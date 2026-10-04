"""H4: контроли без отбора и сквозной прогон H1 → H2 → H3 → H3b → H4 на синтетике."""
import numpy as np
import pandas as pd

from src.cv import h2, h3, h3b, h4, stage5
from src.data import quality, store
from src.stats import trials as T
from tests.fakes import make_bars
from tests.test_stage5_e2e import cfg5  # noqa: F401


def _chain(cfg):
    stage5.run_batch(cfg, tfs=["1h"], slippages=[0.0], schemes=["expanding", "rolling"])
    news = ["NEW1", "NEW2"]
    for i, s in enumerate(news):
        df = make_bars("2021-01-01", 24 * 730, "1h", seed=300 + i)
        df["is_downtime"] = False
        store.write(cfg["paths"]["clean"], s, "1h", df)
    cfg["h2"]["symbols"] = news
    cfg["grids"]["s4_supertrend"]["1h"] = {"n": [24], "m": [3.0, 4.0]}
    allow = lambda c, s, fl: {f.val_start: True for f in fl}
    h2.run(cfg, symbols=news, n_perm=0, tradable_fn=allow,
           data_fn=lambda s: (store.read(cfg["paths"]["clean"], s, "1h"), None))
    cfg["grids"]["s3_donchian"]["4h"] = {"w": [6, 12]}
    cfg["grids"]["s4_supertrend"]["4h"] = {"n": [6], "m": [2.0, 3.0]}
    cfg["grids"]["s4_sma"] = {"4h": {"n": [6], "m": [2.0, 3.0]}}
    cfg["strategies"]["sma_filter"]["bars"]["4h"] = 60
    cfg["selection"]["min_trades_per_year"]["4h"] = 1

    def d4h(s):
        a = quality.aggregate(store.read(cfg["paths"]["clean"], s, "1h"), "4h")
        a = a[a["n"] == 4].drop(columns="n")
        a["is_downtime"] = False
        return a, None

    h3.run(cfg, symbols=h3.symbols_h3(cfg), n_perm=0, tradable_fn=allow, data_fn=d4h)
    cfg["h3b"].update(train_end_exclusive="2023-01-01", val_from="2022-07-01")
    h3b.run(cfg, n_perm=0, tradable_fn=allow, data_fn=d4h)
    cfg["h4"]["train_end_exclusive"] = "2023-01-01"
    return d4h, allow


def test_h4_end_to_end(cfg5):  # noqa: F811
    cfg = cfg5
    d4h, allow = _chain(cfg)
    n_before = int((T.read_trials(cfg["paths"]["trials_log"]).kind == "attempt").sum())
    blocked = pd.Timestamp("2022-07-01", tz="UTC")
    trad = lambda c, s, fl: {f.val_start: not (s == "BBB" and f.val_start == blocked) for f in fl}

    # контроли без отбора: позиция только в разрешённых кварталах валидации
    bh = h4.run_fixed(cfg, kind="bh", slippage=0.0, symbols=["BBB"], tradable_fn=trad, data_fn=d4h)
    p = bh["series"]["pos"]["BBB"]
    q3 = (p.index >= blocked) & (p.index < pd.Timestamp("2022-10-01", tz="UTC"))
    assert (p[q3] == 0).all() and (p[~q3] == 1).all()
    tm = h4.run_fixed(cfg, kind="timing", slippage=0.0, symbols=["AAA"], tradable_fn=allow,
                      data_fn=d4h)
    assert set(np.unique(tm["series"]["pos"]["AAA"])) <= {-1.0, 0.0, 1.0}

    path = h4.run(cfg, n_perm=2, tradable_fn=trad, data_fn=d4h)
    log = T.read_trials(cfg["paths"]["trials_log"])
    att = log[log.kind == "attempt"]
    assert len(att) == n_before + 2
    assert set(att.iloc[-2:]["strategy"]) == {"s4f", "timing"}
    assert len(pd.read_csv(cfg["paths"]["h4_out"] / "noise.csv")) == 2
    text = path.read_text(encoding="utf-8")
    assert f"N = {n_before + 2}" in text and f"в журнале attempt: {n_before + 2}" in text
    assert "Кандидат для test (§5):" in text and "Ledoit–Wolf" in text
    assert "| Buy & hold |" in text and "| Стратегия | 2022 |" in text
    assert "Входы посреди тренда" in text
