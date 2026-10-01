"""Combinatorial Purged CV и PBO (Bailey et al., 2017). Этап 6.

Даёт распределение OOS-Sharpe вместо одной цифры и вероятность
переподгонки процедуры отбора.
"""
from __future__ import annotations

import pandas as pd


def cpcv_splits(index: pd.DatetimeIndex, t1: pd.Series, *, n_groups: int,
                k_test: int, embargo: float) -> list[tuple[pd.Index, pd.Index]]:
    """Все C(n_groups, k_test) разбиений с purging и embargo."""
    raise NotImplementedError


def pbo(is_perf: pd.DataFrame, oos_perf: pd.DataFrame) -> float:
    """PBO: доля разбиений, где лучшая in-sample конфигурация ниже медианы OOS.

    Args:
        is_perf, oos_perf: строки — разбиения, колонки — конфигурации.
    """
    raise NotImplementedError
