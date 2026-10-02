"""Probabilistic и Deflated Sharpe Ratio (Bailey, López de Prado, 2012, 2014).

Все Sharpe — НА БАР (не годовые). Эксцесс — обычный (нормальное = 3).

PSR(SR*) = Φ( (SR − SR*)·√(T − 1) / √(1 − γ₃·SR + (γ₄ − 1)/4·SR²) ).
Ожидаемый максимум N независимых нулевых стратегий с дисперсией Sharpe V:
SR₀ = √V · [(1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))], γ — Эйлера–Маскерони;
при N ≤ 1 SR₀ = 0. DSR = PSR(SR₀).
Свойство: PSR(SR*) = 0.5 ⇔ SR = SR* (не зависит от асимметрии и эксцесса),
поэтому DSR ≥ 0.5 ⇔ SR ≥ SR₀.

N — эффективное число попыток (src.stats.neff), журнал — src.stats.trials.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: float, var_trials: float) -> float:
    """E[max SR] среди N независимых попыток при нулевом истинном SR (на бар)."""
    if n_trials <= 1 or not var_trials > 0:
        return 0.0
    n = float(n_trials)
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(var_trials) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def psr(sr: float, sr_star: float, *, n_obs: int, skew: float, kurt: float) -> float:
    """Вероятность, что истинный SR > sr_star (всё на бар; kurt — обычный)."""
    if n_obs < 2 or np.isnan(sr):
        return np.nan
    den = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if den <= 0:
        return np.nan
    return float(norm.cdf((sr - sr_star) * np.sqrt(n_obs - 1) / np.sqrt(den)))


def dsr(sr: float, *, n_trials: float, var_trials: float, skew: float,
        kurt: float, n_obs: int) -> float:
    """DSR = PSR(E[max SR]) — вероятность, что SR лучшей попытки не результат отбора."""
    return psr(sr, expected_max_sharpe(n_trials, var_trials), n_obs=n_obs, skew=skew, kurt=kurt)


def moments(x: np.ndarray) -> tuple[float, float, float, int]:
    """SR на бар, асимметрия, обычный эксцесс, T — для PSR/DSR."""
    x = np.asarray(x, dtype="float64")
    x = x[~np.isnan(x)]
    t = len(x)
    if t < 3:
        return np.nan, np.nan, np.nan, t
    mu, sd = x.mean(), x.std(ddof=1)
    if sd <= 0:
        return np.nan, np.nan, np.nan, t
    z = (x - mu) / x.std(ddof=0)
    return float(mu / sd), float((z ** 3).mean()), float((z ** 4).mean()), t
