"""Ledoit–Wolf (2008), HAC: разность Sharpe на синтетике."""
import numpy as np
import pandas as pd
import pytest

from src.stats.lw import hac, lw_sharpe_diff


def _pair(rng, t, sr1, sr2, rho, phi=0.0):
    """Два ряда с заданными Sharpe на бар, корреляцией ошибок rho и AR(1) phi."""
    e = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=t)
    if phi:
        for i in range(1, t):
            e[i] += phi * e[i - 1]
        e *= np.sqrt(1 - phi ** 2)
    idx = pd.date_range("2022-01-01", periods=t, freq="4h", tz="UTC")
    return pd.Series(sr1 + e[:, 0], idx), pd.Series(sr2 + e[:, 1], idx)


def test_identical_series_zero():
    rng = np.random.default_rng(0)
    a, _ = _pair(rng, 500, 0.05, 0.0, 0.0)
    d, se, p = lw_sharpe_diff(a, a, periods_per_year=2190)
    assert d == 0 and se == pytest.approx(0, abs=1e-12) and p == 1.0


def test_iid_matches_closed_form():
    """iid нормальные: Var(ΔSR̂) ≈ [2 − 2ρ + ½(SR₁² + SR₂² − 2 SR₁ SR₂ ρ²)] / T."""
    rng = np.random.default_rng(1)
    t, s1, s2, rho = 20000, 0.05, 0.02, 0.6
    a, b = _pair(rng, t, s1, s2, rho)
    _, se, _ = lw_sharpe_diff(a, b, periods_per_year=1)
    v = (2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / t
    assert se == pytest.approx(np.sqrt(v), rel=0.1)


def test_se_matches_monte_carlo_under_autocorrelation():
    """AR(1) φ = 0.3: средний SE ≈ разброс оценок ΔSR; iid-формула занижает."""
    rng = np.random.default_rng(2)
    t, reps = 2000, 200
    est, ses = [], []
    for _ in range(reps):
        a, b = _pair(rng, t, 0.04, 0.0, 0.5, phi=0.3)
        d, se, _ = lw_sharpe_diff(a, b, periods_per_year=1)
        est.append(d)
        ses.append(se)
    mc = float(np.std(est, ddof=1))
    assert np.mean(ses) == pytest.approx(mc, rel=0.2)
    iid = np.sqrt((2 - 2 * 0.5) / t)
    assert np.mean(ses) > 1.2 * iid
    assert np.mean(est) == pytest.approx(0.04, abs=0.01)


def test_size_under_null():
    rng = np.random.default_rng(3)
    rej = [lw_sharpe_diff(*_pair(rng, 1500, 0.03, 0.03, 0.4, phi=0.2),
                          periods_per_year=1)[2] < 0.05 for _ in range(200)]
    assert 0.01 <= np.mean(rej) <= 0.10


def test_power_and_annualization():
    rng = np.random.default_rng(4)
    a, b = _pair(rng, 20000, 0.05, 0.0, 0.3)
    d1, se1, p = lw_sharpe_diff(a, b, periods_per_year=1)
    d2, se2, _ = lw_sharpe_diff(a, b, periods_per_year=2190)
    assert p < 0.01 and d1 > 0
    assert d2 == pytest.approx(d1 * np.sqrt(2190)) and se2 == pytest.approx(se1 * np.sqrt(2190))


def test_alignment_and_kernels():
    rng = np.random.default_rng(5)
    a, b = _pair(rng, 3000, 0.04, 0.01, 0.5, phi=0.2)
    full = lw_sharpe_diff(a, b, periods_per_year=1)
    part = lw_sharpe_diff(a, b.iloc[100:], periods_per_year=1)
    ref = lw_sharpe_diff(a.iloc[100:], b.iloc[100:], periods_per_year=1)
    assert part == pytest.approx(ref) and part[0] != full[0]
    ses = [lw_sharpe_diff(a, b, periods_per_year=1, kernel=k)[1] for k in ("qs", "parzen", "bartlett")]
    assert max(ses) / min(ses) < 1.3
    with pytest.raises(ValueError):
        hac(np.zeros((10, 2)), "boxcar", bandwidth=2.0)
