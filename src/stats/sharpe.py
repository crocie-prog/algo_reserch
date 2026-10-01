"""Sharpe и его стандартная ошибка с учётом автокорреляции (Lo, 2002). rf = 0."""
from __future__ import annotations

import pandas as pd


def sharpe(r: pd.Series, periods_per_year: float) -> float:
    """Годовой Sharpe: mean / std (ddof=1) · √P."""
    raise NotImplementedError


def sharpe_se_lo(r: pd.Series, periods_per_year: float, q: int) -> tuple[float, float]:
    """Годовой Sharpe и его SE по Lo (2002) с поправкой на автокорреляции до лага q.

    Returns:
        (sr, se); 95% ДИ ≈ sr ± 1.96·se.
    """
    raise NotImplementedError
