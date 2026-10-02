"""Sharpe и его стандартная ошибка с учётом автокорреляции (Lo, 2002). rf = 0.

SR на бар = mean / std (ddof=1). Годовой SR = SR_бар · √P — простая
аннуализация, одна и та же во всём проекте (поправку η(q) Lo для
аннуализации не применяем, чтобы сравнения были однородны).

SE: GMM/дельта-метод по моментам m1 = r − μ, m2 = (r − μ)² − σ² с HAC-оценкой
Newey–West (веса Бартлетта, q лагов; по умолчанию q = ⌊T^{1/4}⌋). При iid
сводится к √((1 + SR²/2)/T) (без поправки на асимметрию/эксцесс — её даёт HAC
через момент m2). Годовой SE = SE_бар · √P.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(r: pd.Series | np.ndarray, periods_per_year: float) -> float:
    """Годовой Sharpe: mean / std (ddof=1) · √P."""
    x = np.asarray(r, dtype="float64")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan


def _newey_west(m: np.ndarray, q: int) -> np.ndarray:
    """Долгосрочная ковариация моментов (T × k), веса Бартлетта."""
    t = len(m)
    s = m.T @ m / t
    for lag in range(1, q + 1):
        w = 1.0 - lag / (q + 1)
        g = m[lag:].T @ m[:-lag] / t
        s += w * (g + g.T)
    return s


def sharpe_se_lo(r: pd.Series | np.ndarray, periods_per_year: float,
                 q: int | None = None) -> tuple[float, float]:
    """Годовой Sharpe и его SE по Lo (2002) с HAC до лага q.

    Returns:
        (sr_ann, se_ann); 95% ДИ ≈ sr ± 1.96·se; t = sr/se.
    """
    x = np.asarray(r, dtype="float64")
    x = x[~np.isnan(x)]
    t = len(x)
    if t < 3:
        return np.nan, np.nan
    q = int(np.floor(t ** 0.25)) if q is None else int(q)
    mu = x.mean()
    var = x.var(ddof=0)
    if var <= 0:
        return np.nan, np.nan
    sd = np.sqrt(var)
    m = np.column_stack([x - mu, (x - mu) ** 2 - var])
    omega = _newey_west(m, q)
    # SR = μ / σ = μ / √v; градиент по (μ, v)
    g = np.array([1.0 / sd, -mu / (2.0 * var * sd)])
    se_bar = float(np.sqrt(g @ omega @ g / t))
    k = np.sqrt(periods_per_year)
    return float(mu / sd * k), se_bar * k
