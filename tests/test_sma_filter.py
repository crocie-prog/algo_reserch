"""H4: фильтр направления SMA, S4 + SMA, тайминг SMA."""
import numpy as np
import pandas as pd
import pytest

from src.strategies import s4_sma, s4_supertrend, sma_filter, sma_timing
from tests.fakes import make_bars


@pytest.fixture
def df():
    d = make_bars("2021-01-01", 6 * 700, "4h", seed=7)
    d["is_downtime"] = False
    return d


def test_allowed_definition_and_warmup(df):
    a = sma_filter.allowed(df, sma_bars=60)
    sma = df["close"].rolling(60).mean()
    assert not (a["long"].iloc[:59].any() or a["short"].iloc[:59].any())      # прогрев
    ok = sma.notna()
    assert (a["long"][ok] == (df["close"] > sma)[ok]).all()
    assert (a["short"][ok] == (df["close"] < sma)[ok]).all()
    assert not (a["long"] & a["short"]).any()


def test_mask_matches_s4(df):
    p4 = s4_supertrend.signal(df, n=6, m=2.0)
    pf = s4_sma.signal(df, n=6, m=2.0, sma_bars=60)
    a = sma_filter.allowed(df, sma_bars=60)
    allow = np.where(p4 > 0, a["long"], np.where(p4 < 0, a["short"], False))
    assert np.array_equal(pf.to_numpy(), np.where(allow, p4, 0.0))
    assert (pf != 0).sum() < (p4 != 0).sum()                    # фильтр что-то запрещает
    assert ((pf > 0) & (df["close"] <= a["sma"])).sum() == 0
    assert ((pf < 0) & (df["close"] >= a["sma"])).sum() == 0


def test_causal(df):
    """Изменение будущих баров не меняет прошлые позиции."""
    cut = 3000
    d2 = df.copy()
    d2.iloc[cut:, :4] *= 1.5
    for f in (lambda d: s4_sma.signal(d, n=6, m=2.0, sma_bars=60),
              lambda d: sma_timing.signal(d, sma_bars=60)):
        assert f(df).iloc[:cut].equals(f(d2).iloc[:cut])


def test_shift_minus_one_is_suspicious(df):
    """Санити: тайминг, сдвинутый на −1 бар (заглядывание), даёт явно лучший результат."""
    r = df["close"].pct_change().fillna(0)
    p = sma_timing.signal(df, sma_bars=60)
    honest = (p.shift(1) * r).sum()
    cheat = (p.shift(-1).shift(1) * r).sum()
    assert cheat > honest


def test_downtime_freezes_permissions(df):
    d = df.copy()
    i = 2000
    d.iloc[i:i + 3, d.columns.get_loc("is_downtime")] = True
    a = sma_filter.allowed(d, sma_bars=60)
    assert (a["long"].iloc[i:i + 3] == a["long"].iloc[i - 1]).all()
    assert (a["short"].iloc[i:i + 3] == a["short"].iloc[i - 1]).all()


def test_timing_and_config(df, cfg):
    p = sma_timing.signal(df, sma_bars=60)
    assert set(np.unique(p)) <= {-1.0, 0.0, 1.0} and (p.iloc[:59] == 0).all()
    assert s4_sma.config_params("4h", cfg) == {"sma_bars": 1200}
    assert s4_sma.warmup(n=6, sma_bars=1200) == 1200 and s4_sma.warmup(n=500, sma_bars=10) == 1500
