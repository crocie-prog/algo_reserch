"""Веса портфеля. Оцениваются только по прошлому (окно до момента ребалансировки).

Порядок внедрения: EW, IV → RP → MS (последним и со скепсисом, DeMiguel et al., 2009).
Схема фиксируется заранее, не выбирается по результату.
"""
from __future__ import annotations

import pandas as pd


def weights(train_returns: pd.DataFrame, scheme: str) -> pd.Series:
    """Веса long-only, Σw = 1.

    - EW: 1/N;
    - IV: 1/σ_i, нормированные;
    - RP: равные вклады в риск;
    - MS: максимум Sharpe (mean-variance).

    Args:
        train_returns: доходности компонент до момента ребалансировки.
    """
    raise NotImplementedError
