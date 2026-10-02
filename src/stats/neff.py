"""Эффективное число попыток по собственным значениям корреляционной матрицы.

Основной метод (предрегистрация §8): N_eff = число наибольших собственных
значений, объясняющих share (0.95) следа матрицы. Participation ratio
(Σλ)²/Σλ² — нижняя граница (занижает N, для DSR неконсервативна), выводится
для справки.
"""
from __future__ import annotations

import numpy as np


def _eig(corr: np.ndarray) -> np.ndarray:
    c = np.asarray(corr, dtype="float64")
    c = np.nan_to_num((c + c.T) / 2.0)
    lam = np.clip(np.linalg.eigvalsh(c), 0.0, None)
    return np.sort(lam)[::-1]


def effective_number(corr: np.ndarray, share: float = 0.95) -> int:
    """Число собственных значений, дающих ≥ share следа."""
    lam = _eig(corr)
    if lam.sum() <= 0:
        return 1
    return int(np.searchsorted(np.cumsum(lam) / lam.sum(), share - 1e-12) + 1)


def participation_ratio(corr: np.ndarray) -> float:
    """(Σλ)² / Σλ²."""
    lam = _eig(corr)
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if (lam ** 2).sum() > 0 else 1.0


def corr_of_columns(x: np.ndarray) -> np.ndarray:
    """Корреляция столбцов (T × k); столбцы нулевой дисперсии отбрасываются заранее."""
    x = np.asarray(x, dtype="float64")
    if x.shape[1] == 1:
        return np.ones((1, 1))
    return np.corrcoef(x, rowvar=False)
