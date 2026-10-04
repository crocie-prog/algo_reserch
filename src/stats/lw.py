"""Тест разности Sharpe Ledoit–Wolf (2008) с HAC-оценкой ковариации.

Delta-метод для ΔSR = SR_1 − SR_2 по моментам ν = (μ_1, μ_2, γ_1, γ_2),
γ_i = E r_i²: SR_i = μ_i / √(γ_i − μ_i²). Ковариация моментов — HAC
(Andrews, 1991): ядро QS (по умолчанию), Parzen или Bartlett, полоса —
автоматическая по AR(1)-аппроксимации каждого ряда моментов (веса 1),
поправка малой выборки T / (T − 4), как у Ledoit–Wolf; лаги QS
отсекаются на 50·S (вес ядра там < 10⁻³). Префильтрация
(prewhitening) не используется.

Ряды выравниваются по индексу (общие наблюдения без NaN). Sharpe и SE —
годовые (× √periods_per_year); p-value — двусторонний, нормальное
приближение (не бутстрап).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

_BW = {"qs": (1.3221, 2), "parzen": (2.6614, 2), "bartlett": (1.1447, 1)}


def _kernel(x: np.ndarray, kind: str) -> np.ndarray:
    x = np.abs(np.asarray(x, dtype="float64"))
    if kind == "qs":
        z = 6.0 * math.pi * x / 5.0
        with np.errstate(divide="ignore", invalid="ignore"):
            k = 25.0 / (12.0 * math.pi ** 2 * x ** 2) * (np.sin(z) / z - np.cos(z))
        return np.where(x == 0, 1.0, k)
    if kind == "parzen":
        return np.where(x <= 0.5, 1 - 6 * x ** 2 + 6 * x ** 3,
                        np.where(x <= 1, 2 * (1 - x) ** 3, 0.0))
    if kind == "bartlett":
        return np.clip(1 - x, 0.0, None)
    raise ValueError(f"неизвестное ядро: {kind}")


def andrews_bandwidth(y: np.ndarray, kind: str = "qs") -> float:
    """Автоматическая полоса Andrews (1991) по AR(1) для каждой колонки y (T × k)."""
    c, q = _BW[kind]
    t = len(y)
    num = den = 0.0
    for a in range(y.shape[1]):
        u = y[:, a]
        x0, x1 = u[:-1], u[1:]
        d = float(x0 @ x0)
        rho = float(x0 @ x1) / d if d > 0 else 0.0
        rho = min(max(rho, -0.97), 0.97)
        s2 = float(np.mean((x1 - rho * x0) ** 2))
        if q == 2:
            num += 4 * rho ** 2 * s2 ** 2 / (1 - rho) ** 8
        else:
            num += 4 * rho ** 2 * s2 ** 2 / ((1 - rho) ** 6 * (1 + rho) ** 2)
        den += s2 ** 2 / (1 - rho) ** 4
    alpha = num / den if den > 0 else 0.0
    return float(c * (alpha * t) ** (1.0 / (2 * q + 1))) if alpha > 0 else 0.0


def hac(y: np.ndarray, kind: str = "qs", bandwidth: float | None = None) -> np.ndarray:
    """HAC-оценка долгосрочной ковариации центрированных моментов y (T × k)."""
    t, k = y.shape
    s = andrews_bandwidth(y, kind) if bandwidth is None else float(bandwidth)
    psi = y.T @ y / t
    if s > 0:
        # у Parzen и Bartlett вес 0 при лаге > S; хвост QS ~ 1/x² — отсечка на 50·S
        max_lag = min(t - 1, int(math.ceil(50 * s)) if kind == "qs" else int(math.floor(s)))
        for j in range(1, max_lag + 1):
            w = float(_kernel(np.array([j / s]), kind)[0])
            if w == 0.0:
                continue
            g = y[j:].T @ y[:-j] / t
            psi += w * (g + g.T)
    return psi * t / (t - k) if t > k else psi


def lw_sharpe_diff(r1: pd.Series, r2: pd.Series, *, periods_per_year: float,
                   kernel: str = "qs") -> tuple[float, float, float]:
    """(ΔSR годовой, SE, двусторонний p-value). Ряды выравниваются по индексу."""
    j = pd.concat([pd.Series(r1), pd.Series(r2)], axis=1, join="inner").dropna()
    x = j.to_numpy(dtype="float64")
    t = len(x)
    if t < 10:
        return np.nan, np.nan, np.nan
    mu = x.mean(axis=0)
    gam = (x ** 2).mean(axis=0)
    var = gam - mu ** 2
    if (var <= 0).any():
        return np.nan, np.nan, np.nan
    sr = mu / np.sqrt(var)
    delta = float(sr[0] - sr[1])
    y = np.column_stack([x[:, 0] - mu[0], x[:, 1] - mu[1],
                         x[:, 0] ** 2 - gam[0], x[:, 1] ** 2 - gam[1]])
    psi = hac(y, kernel)
    v32 = var ** 1.5
    grad = np.array([gam[0] / v32[0], -gam[1] / v32[1],
                     -mu[0] / (2 * v32[0]), mu[1] / (2 * v32[1])])
    se = float(np.sqrt(max(grad @ psi @ grad, 0.0) / t))
    k = math.sqrt(periods_per_year)
    if se == 0.0:
        return delta * k, 0.0, 1.0 if delta == 0 else 0.0
    p = float(2 * norm.sf(abs(delta) / se))
    return delta * k, se * k, p
