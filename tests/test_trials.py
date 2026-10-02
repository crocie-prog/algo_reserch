"""Журнал попыток: дописывание, типы, train-записи по walk-forward."""
import pandas as pd
import pytest

from src.stats import trials as T
from tests.test_walkforward import _folds, _grid, random_walk
from src.cv.walkforward import walk_forward


def test_append_only_and_kinds(tmp_path):
    p = tmp_path / "trials_log.csv"
    T.log_trial(p, kind="attempt", variant="v", strategy="s1", tf="1h", symbol="EW",
                params={"rule": "ensemble"}, sharpe=0.1, n_obs=10)
    T.log_trial(p, kind="sensitivity", variant="v2", strategy="s1", tf="1h", symbol="EW",
                params={}, sharpe=0.0, n_obs=10)
    t = T.read_trials(p)
    assert len(t) == 2 and list(t.columns) == T.COLUMNS
    assert T.n_trials(p, kind="attempt") == 1
    with pytest.raises(ValueError):
        T.log_trial(p, kind="other", variant="v", strategy="s", tf="1h", symbol="X",
                    params={}, sharpe=0, n_obs=1)
    assert len(T.read_trials(p)) == 2


def test_train_rows_from_walk_forward(cfg, tmp_path):
    cfg["universe"]["warmup_only_until"] = {}
    cfg["costs"]["fee_per_side"] = 0.0
    cfg["costs"]["include_funding"] = False
    cfg["selection"]["min_trades_per_year"]["1h"] = 1
    df = random_walk(7)
    gd = _grid(df, cfg)
    wf = walk_forward(gd, df, None, _folds(df), cfg, slippage=0.0, periods_per_year=8760)
    rows = T.train_rows(wf, gd, "1h|expanding|slip=0")
    assert len(rows) == sum(s.n_pass for s in wf.selections)
    assert (rows["kind"] == "train").all() and (rows["n_obs"] > 0).all()
    p = tmp_path / "t.csv"
    T.log_rows(p, rows)
    T.log_rows(p, rows)
    assert len(T.read_trials(p)) == 2 * len(rows)
