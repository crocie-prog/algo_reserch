"""Deflated Sharpe Ratio (Bailey, López de Prado, 2014).

Число попыток N берётся из журнала (src.stats.trials) — честно, включая
неудачные варианты дизайна.
"""
from __future__ import annotations


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """E[max SR] среди N независимых попыток при нулевом истинном SR."""
    raise NotImplementedError


def dsr(sr: float, *, n_trials: int, var_trials: float, skew: float,
        kurt: float, n_obs: int) -> float:
    """Вероятность, что истинный SR > E[max SR] (SR в единицах за бар)."""
    raise NotImplementedError
