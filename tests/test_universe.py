"""Шаг 6 этапа 1: universe.csv."""
import pandas as pd

from src.data import clean, store, universe
from tests.fakes import FakeBybit, make_bars


def test_universe_table(cfg):
    cfg["universe"]["symbols"] = ["X"]
    m1 = make_bars("2021-10-15", 120, "1min")
    store.write(cfg["paths"]["raw"], "X", "1m", m1)
    clean.build_clean(cfg, "X", "1m")
    ex = FakeBybit("2026-10-01", instruments={"X": {"symbol": "X", "status": "Trading",
                                                    "launchTime": "1634256000000",
                                                    "fundingInterval": 480}})
    df = universe.build_universe(cfg, ex)
    r = pd.read_csv(cfg["paths"]["meta"]).iloc[0]
    assert r["status"] == "Trading" and r["funding_interval_minutes"] == 480
    assert pd.Timestamp(r["first_bar_1m"]) == m1.index[0]
    assert pd.Timestamp(r["last_bar_1m"]) == m1.index[-1]
    assert r["fixed_on"] == "2021-10-15" and "не обрезка" in r["note"]
    assert pd.isna(r["first_bar_1h"])
