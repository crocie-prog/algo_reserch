"""S2 Bollinger: эквивалентность S1(z_exit=0), полосы, параметры, look-ahead."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.checks import truncation_check
from src.strategies import s1_zscore as s1
from src.strategies import s2_bollinger as s2
from src.strategies._common import downtime_mask, run_state_machine
from tests.fakes import make_bars


def _bands_rule(df, w, k):
    """Независимая реализация правила Боллинджера по полосам (close ≤ lower …)."""
    b = s2.bands(df, w=w, k=k)
    c = df["close"]
    ok = b["mid"].notna().to_numpy() & (b["upper"] > b["lower"]).to_numpy()
    el = ok & (c <= b["lower"]).to_numpy()
    es = ok & (c >= b["upper"]).to_numpy()
    xl = ok & (c >= b["mid"]).to_numpy()
    xs = ok & (c <= b["mid"]).to_numpy()
    return pd.Series(run_state_machine(el, es, xl, xs, frozen=downtime_mask(df)), index=df.index)


def _synthetic():
    df = make_bars("2023-01-01", 5000, "1h", seed=31)
    down = np.zeros(5000, bool)
    down[2000:2030] = True
    df["is_downtime"] = down
    return df


@pytest.mark.parametrize("w,k", [(24, 2.0), (72, 1.5), (168, 2.5)])
def test_equivalent_to_s1_zexit0_synthetic(w, k):
    df = _synthetic()
    a = s2.signal(df, w=w, k=k)
    pd.testing.assert_series_equal(a, s1.signal(df, w=w, z_entry=k, z_exit=0.0))
    # и независимое правило по полосам даёт то же самое
    assert (a.to_numpy() == _bands_rule(df, w, k).to_numpy()).all()


@pytest.mark.data
def test_equivalent_real_btc_1h():
    try:
        from src.data.load import load
        df = load("BTCUSDT", "1h")
    except Exception:
        pytest.skip("нет clean BTCUSDT 1h")
    for w in (24, 72, 168):
        a = s2.signal(df, w=w, k=2.0)
        assert (a.to_numpy() == _bands_rule(df, w, 2.0).to_numpy()).all()
        pd.testing.assert_series_equal(a, s1.signal(df, w=w, z_entry=2.0, z_exit=0.0))
    truncation_check(s2.signal, df, dict(w=72, k=2.0))


def test_bands_and_target():
    df = _synthetic()
    b = s2.bands(df, w=24, k=2.0)
    assert np.allclose((b["upper"] - b["mid"]).dropna(), (b["mid"] - b["lower"]).dropna())
    assert b[df["is_downtime"]].isna().all().all()
    t = s2.target_move(df, w=24, k=2.0)
    assert np.allclose(t.dropna(), ((b["upper"] - b["mid"]) / df["close"]).dropna())


@pytest.mark.parametrize("kw", [dict(w=24, k=0.0), dict(w=1, k=2.0), dict(w=24, k=-1.0)])
def test_param_validation(kw):
    with pytest.raises(ValueError):
        s2.signal(_synthetic(), **kw)


def test_truncation_synthetic():
    truncation_check(s2.signal, _synthetic(), dict(w=48, k=2.0))
