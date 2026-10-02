"""Сквозной прогон этапа 5 на синтетике: варианты, журнал, шум, вердикты."""
import numpy as np
import pandas as pd
import pytest

from src.cv import report, stage5
from src.data import store
from src.stats import trials as T
from tests.fakes import make_bars


@pytest.fixture
def cfg5(cfg):
    cfg["universe"]["symbols"] = ["AAA", "BBB", "CCC"]
    cfg["universe"]["warmup_only_until"] = {"AAA": "2021-01-08"}
    st = cfg["stage5"]
    st.update(train_end_exclusive="2023-01-01", tfs=["1h"], slippage=[0.0, 0.0005],
              permutations=2, s6_train_from="2022-04-01")
    cfg["periods"]["train_end"] = "2022-12-31"
    cfg["selection"]["min_trades_per_year"]["1h"] = 1
    cfg["grids"] = {
        "s1_zscore": {"1h": {"w": [24, 48], "z_entry": [1.5, 2.0], "z_exit": [0.0]}},
        "s3_donchian": {"1h": {"w": [24, 48]}},
        "s4_supertrend": {"1h": {"n": [24], "m": [3.0, 4.0]}},
        "s5_pivot": {"1h": {}},
        "s6_vwap": {"1h": {"k": [1.5, 2.5], "e": [0.5]}},
    }
    for i, sym in enumerate(cfg["universe"]["symbols"]):
        df = make_bars("2021-01-01", 24 * 730, "1h", seed=200 + i)
        df["is_downtime"] = False
        store.write(cfg["paths"]["clean"], sym, "1h", df)
        f = pd.DataFrame({"funding_rate": 1e-4},
                         index=pd.date_range("2021-01-01 08:00", periods=3 * 730, freq="8h", tz="UTC"))
        store.write(cfg["paths"]["raw"], sym, "funding", f)
    return cfg


def test_end_to_end(cfg5):
    vids = stage5.run_batch(cfg5, tfs=["1h"], slippages=[0.0, 0.0005],
                            schemes=["expanding", "rolling"])
    assert set(vids) == {"1h_expanding_slip0", "1h_rolling_slip0", "1h_expanding_slip0_s6from2022",
                         "1h_rolling_slip0_s6from2022", "1h_expanding_slip0.0005",
                         "1h_rolling_slip0.0005", "1h_expanding_slip0.0005_s6from2022",
                         "1h_rolling_slip0.0005_s6from2022"}
    root = stage5.out_root(cfg5)
    folds = pd.read_csv(root / "1h_expanding_slip0" / "folds.csv")
    assert set(folds.groupby(["symbol", "strategy"]).size()) == {3, 4}   # AAA — 3 фолда (usable_from)
    assert (root / "1h_expanding_slip0" / "ew" / "s1_zscore_plateau.parquet").exists()

    log = T.read_trials(cfg5["paths"]["trials_log"])
    att = log[log.kind == "attempt"]
    assert len(att) == 22                                       # как в предрегистрации §8
    assert set(log.kind) == {"train", "attempt", "sensitivity"}
    assert (log.loc[log.kind == "sensitivity", "variant"].str.contains("slip0.0005")).all()

    noise = stage5.noise_baseline(cfg5, n_perm=2)
    assert len(noise) == 2 * 3 * 5 and noise["share_no_trade"].between(0, 1).all()

    path = report.verdicts(cfg5)
    text = path.read_text(encoding="utf-8")
    for s in stage5.STRATEGIES:
        assert s in text
    assert "Уровень 2" in text and "шума" in text
    oos = pd.read_csv(root / "oos.csv")
    assert {"noise_share_mean", "noise_share_p5", "share_no_trade"} <= set(oos.columns)
    v = pd.read_csv(root / "verdicts.csv")
    assert set(v["verdict"]) <= {"Значимо", "Кандидат", "Нет", "недостаточно данных"}
    assert str(path).startswith(str(cfg5["paths"]["stage5_verdicts"].parent))


def test_level2_idle_attempts_form_one_cluster():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=2000, freq="1h", tz="UTC")
    series = {f"a{i}": pd.Series(rng.normal(0, 0.01, 2000), index=idx) for i in range(3)}
    base = report.level2_from_series(series, [], 0.95)
    with_idle = report.level2_from_series(series, ["z1", "z2", "z3"], 0.95)
    assert with_idle["n_eff"] == base["n_eff"] + 1
    assert with_idle["n_attempts"] == 6 and with_idle["n_idle"] == 3
    assert with_idle["var_sr_bar"] == pytest.approx(base["var_sr_bar"])
    assert with_idle["sr0_bar"] > base["sr0_bar"]
    only_idle = report.level2_from_series({}, ["z"], 0.95)
    assert only_idle["n_eff"] == 1 and only_idle["sr0_bar"] == 0.0
