"""Сквозной прогон H2 на синтетике: журнал, N сквозное, фильтр торгуемости, вердикт."""
import pandas as pd

from src.cv import h2, stage5
from src.stats import trials as T
from tests.test_stage5_e2e import cfg5  # noqa: F401  (фикстура H1 на синтетике)
from src.data import store
from tests.fakes import make_bars


def test_h2_end_to_end(cfg5):  # noqa: F811
    cfg = cfg5
    stage5.run_batch(cfg, tfs=["1h"], slippages=[0.0], schemes=["expanding", "rolling"])
    n_h1 = int((T.read_trials(cfg["paths"]["trials_log"]).kind == "attempt").sum())
    syms = ["NEW1", "NEW2"]
    for i, s in enumerate(syms):
        df = make_bars("2021-01-01", 24 * 730, "1h", seed=300 + i)
        df["is_downtime"] = False
        store.write(cfg["paths"]["clean"], s, "1h", df)
    cfg["h2"]["symbols"] = syms
    cfg["grids"]["s4_supertrend"] = {"1h": {"n": [24], "m": [3.0, 4.0]}}
    blocked = pd.Timestamp("2022-07-01", tz="UTC")
    trad = lambda c, s, fl: {f.val_start: not (s == "NEW2" and f.val_start == blocked) for f in fl}
    data = lambda s: (store.read(cfg["paths"]["clean"], s, "1h"), None)
    path = h2.run(cfg, symbols=syms, n_perm=2, tradable_fn=trad, data_fn=data)

    log = T.read_trials(cfg["paths"]["trials_log"])
    att = log[log.kind == "attempt"]
    assert len(att) == n_h1 + 1 and att.iloc[-1]["symbol"] == "EW2"
    assert (log.loc[log.variant.str.startswith("h2_") & (log.kind == "sensitivity"), "variant"]
            .str.contains("slip0.0005")).any()
    out = cfg["paths"]["h2_out"]
    f0 = pd.read_csv(out / "h2_1h_expanding_best_slip0" / "folds.csv")
    row = f0[(f0.symbol == "NEW2") & (f0.val_start.str.startswith("2022-07-01"))]
    assert not row.liquidity_ok.iloc[0] and not row.trade_best.iloc[0]
    assert len(pd.read_csv(out / "noise.csv")) == 2
    text = path.read_text(encoding="utf-8")
    assert "**Вердикт:" in text and f"N = {n_h1 + 1}" in text and "| 2022 |" in text
