"""Экономическая ценность: certainty equivalent (mean-variance).

CE = r − (γ/2)·σ², годовые: r = mean · P, σ² = var · P; γ из config (3, 5, 10).
Одно определение во всём проекте.
"""
from __future__ import annotations

import pandas as pd


def ce(r: pd.Series, *, gamma: float, periods_per_year: float) -> float:
    """Годовой CE ряда простых доходностей."""
    raise NotImplementedError


def delta_ce(r_new: pd.Series, r_base: pd.Series, *, gamma: float,
             periods_per_year: float) -> float:
    """ΔCE — сколько в год инвестор заплатил бы за переход base → new."""
    raise NotImplementedError
