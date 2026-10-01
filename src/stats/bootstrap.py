"""Стационарный бутстрап (Politis, Romano, 1994) для ΔSharpe. Не iid.

Длина блока — Politis, White (2004) с поправкой Patton et al. (2009).
Парный: одинаковые индексы для обеих серий. p-value — при H0: ΔSR = 0
(распределение центрируется на наблюдаемой разности), а не как доля Δ ≤ 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def optimal_block_length(r: pd.Series) -> float:
    """Средняя длина блока по Politis–White."""
    raise NotImplementedError


def stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Индексы одной бутстрап-выборки: блоки геометрической длины, циклически."""
    raise NotImplementedError


def stationary_bootstrap_diff(r1: pd.Series, r2: pd.Series, *, periods_per_year: float,
                              n_boot: int, seed: int,
                              mean_block: float | None = None) -> dict[str, float]:
    """ΔSR, ДИ (перцентильный), p-value при H0, использованная длина блока."""
    raise NotImplementedError
