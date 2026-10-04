"""H3b: критерий §4 и сквозной прогон H1 → H2 → H3 → H3b на синтетике."""
import pandas as pd

from src.cv import h2, h3, h3b, stage5
from src.data import quality, store
from src.stats import trials as T
from tests.fakes import make_bars
from tests.test_stage5_e2e import cfg5  # noqa: F401


def _p(rows):
    return pd.DataFrame(rows, columns=["symbol", "oos_sharpe", "traded"])


def test_criterion():
    idx = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
    up = pd.Series([0.002, -0.001] * 50, index=idx)
    dn = -up
    good = _p([("A", 1.0, True), ("B", 0.5, True), ("C", float("nan"), False)])
    assert h3b.criterion(up, up, good, ppy=2190, min_traded=2)[0]
    assert not h3b.criterion(up, dn, good, ppy=2190, min_traded=2)[0]          # знак при slippage
    assert not h3b.criterion(up, up, _p([("A", 1.0, True), ("B", -0.1, True),
                                         ("C", 2.0, True)]), ppy=2190, min_traded=2)[0]
    one = _p([("A", 1.0, True), ("B", float("nan"), False), ("C", float("nan"), False)])
    ok, c = h3b.criterion(up, up, one, ppy=2190, min_traded=2)
    assert not ok and c["n_traded"] == 1                                       # отказ на 2 из 3


def test_h3b_end_to_end(cfg5):  # noqa: F811
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
    h2.run(cfg, symbols=news, n_perm=0, tradable_fn=allow,
           data_fn=lambda s: (store.read(cfg["paths"]["clean"], s, "1h"), None))
    cfg["grids"]["s3_donchian"]["4h"] = {"w": [6, 12]}
    cfg["grids"]["s4_supertrend"]["4h"] = {"n": [6], "m": [2.0, 3.0]}
    cfg["selection"]["min_trades_per_year"]["4h"] = 1

    def d4h(s):
        a = quality.aggregate(store.read(cfg["paths"]["clean"], s, "1h"), "4h")
        a = a[a["n"] == 4].drop(columns="n")
        a["is_downtime"] = False
        return a, None

    h3.run(cfg, symbols=h3.symbols_h3(cfg), n_perm=0, tradable_fn=allow, data_fn=d4h)
    n_before = int((T.read_trials(cfg["paths"]["trials_log"]).kind == "attempt").sum())

    cfg["h3b"].update(train_end_exclusive="2023-01-01", val_from="2022-07-01")
    path = h3b.run(cfg, n_perm=2, tradable_fn=allow, data_fn=d4h)

    log = T.read_trials(cfg["paths"]["trials_log"])
    assert int((log.kind == "attempt").sum()) == n_before + 1
    ft = pd.read_csv(cfg["paths"]["h3b_out"] / "target_slip0" / "folds.csv")
    assert set(ft.val_start.str[:7]) == {"2022-07", "2022-10"} and ft.topk.notna().all()
    pr = pd.read_csv(cfg["paths"]["h3b_out"] / "target_slip0" / "pairs.csv")
    assert list(pr.symbol) == ["AAA", "BBB", "CCC"]
    assert len(pd.read_csv(cfg["paths"]["h3b_out"] / "noise.csv")) == 2
    text = path.read_text(encoding="utf-8")
    assert "**Вердикт:" in text and f"N = {n_before + 1}" in text and "(H1–H3) + 1" in text
    assert "| 2022Q3 |" in text and "Лонг-сторона" in text and "10 пар H2" in text
