"""Тест разности Sharpe Ledoit–Wolf (2008) с HAC-оценкой ковариации.

Delta-метод для ΔSR = SR_1 − SR_2 по моментам (μ_1, μ_2, E r_1², E r_2²);
ковариация моментов — HAC (ядро QS или Parzen, полоса — Andrews, 1991).
"""
from __future__ import annotations

import pandas as pd


def lw_sharpe_diff(r1: pd.Series, r2: pd.Series, *, periods_per_year: float,
                   kernel: str = "qs") -> tuple[float, float, float]:
    """(ΔSR годовой, SE, двусторонний p-value). Ряды выравниваются по индексу."""
    raise NotImplementedError
