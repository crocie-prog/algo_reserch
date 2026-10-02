"""Описательный прогон: только частотные величины, окно от usable_from."""
import pandas as pd

from src.backtest import describe as d
from src.data import store
from tests.fakes import make_bars


def test_describe_no_return_metrics_and_usable_from(cfg):
    cfg["universe"]["symbols"] = ["X"]
    cfg["universe"]["warmup_only_until"] = {"X": "2023-01-11"}
    df = make_bars("2023-01-01", 24 * 30, "1h", seed=2)
    store.write(cfg["paths"]["clean"], "X", "1h", df)
    tab = d.describe("s1_zscore", [{"w": 24, "z_entry": 1.5, "z_exit": 0.3}], tf="1h", cfg=cfg)
    assert tab["from"].iloc[0] == pd.Timestamp("2023-01-11", tz="UTC")
    forbidden = {"sharpe", "ann_return", "equity", "net", "gross", "mdd"}
    assert not forbidden & set(tab.columns)
    assert tab["n_trades"].iloc[0] > 0
