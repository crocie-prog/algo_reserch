"""Статистика: SE Sharpe по Lo, ожидаемый максимум, PSR/DSR, N_eff."""
import numpy as np
import pytest

from src.stats.dsr import dsr, expected_max_sharpe, moments, psr
from src.stats.neff import effective_number, participation_ratio
from src.stats.sharpe import sharpe, sharpe_se_lo


def test_lo_se_iid_matches_closed_form():
    rng = np.random.default_rng(0)
    t = 200_000
    x = rng.normal(0.01, 1.0, t)
    sr, se = sharpe_se_lo(x, periods_per_year=1, q=0)
    assert sr == pytest.approx(sharpe(x, 1), rel=1e-3)
    assert se == pytest.approx(np.sqrt((1 + sr ** 2 / 2) / t), rel=0.03)


def test_lo_se_grows_with_positive_autocorrelation():
    rng = np.random.default_rng(1)
    t = 100_000
    e = rng.normal(size=t)
    x = np.empty(t)
    x[0] = e[0]
    for i in range(1, t):
        x[i] = 0.5 * x[i - 1] + e[i]
    x += 0.02
    _, se_iid = sharpe_se_lo(x, 1, q=0)
    _, se_hac = sharpe_se_lo(x, 1, q=20)
    assert se_hac > 1.5 * se_iid                       # для AR(0.5) ≈ √3


def test_psr_half_at_equality_and_monotone():
    assert psr(0.05, 0.05, n_obs=1000, skew=-1.0, kurt=8.0) == pytest.approx(0.5)
    assert psr(0.06, 0.05, n_obs=1000, skew=0, kurt=3) > 0.5
    assert psr(0.04, 0.05, n_obs=1000, skew=0, kurt=3) < 0.5


def test_expected_max_matches_monte_carlo():
    rng = np.random.default_rng(2)
    for n in (10, 100):
        sims = rng.normal(0, 0.1, size=(20_000, n)).max(axis=1).mean()
        assert expected_max_sharpe(n, 0.01) == pytest.approx(sims, rel=0.05)
    assert expected_max_sharpe(1, 0.01) == 0.0


def test_dsr_equals_psr_at_expected_max():
    sr0 = expected_max_sharpe(20, 0.0004)
    assert dsr(sr0, n_trials=20, var_trials=0.0004, skew=0, kurt=3, n_obs=500) == pytest.approx(0.5)


def test_moments():
    rng = np.random.default_rng(3)
    x = rng.normal(0.1, 1.0, 100_000)
    sr, sk, ku, t = moments(x)
    assert sr == pytest.approx(0.1, abs=0.01) and abs(sk) < 0.05 and ku == pytest.approx(3, abs=0.1)


def _block_corr(sizes, rho_in=1.0):
    n = sum(sizes)
    c = np.zeros((n, n))
    i = 0
    for s in sizes:
        c[i:i + s, i:i + s] = rho_in
        i += s
    np.fill_diagonal(c, 1.0)
    return c


def test_effective_number_blocks():
    assert effective_number(np.eye(10)) == 10                 # независимые: 95% → 10
    assert effective_number(_block_corr([5, 5, 5])) == 3      # дубликаты: число различных
    assert participation_ratio(_block_corr([5, 5, 5])) == pytest.approx(3.0)
    c = _block_corr([4, 4], rho_in=0.9)
    assert participation_ratio(c) <= effective_number(c)
