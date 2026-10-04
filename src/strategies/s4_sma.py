"""S4 + фильтр направления SMA (гипотеза H4).

Позиция S4 (src.strategies.s4_supertrend) маскируется фильтром
src.strategies.sma_filter: лонг только при close > SMA, шорт только при
close < SMA, иначе 0. Маска: при возврате разрешения позиция S4
восстанавливается посреди тренда, без нового сигнала S4. Сетка — n, m S4;
sma_bars — из config, не оптимизируется. cost_to_band — по полосе S4.
"""
from __future__ import annotations

import pandas as pd

from src.strategies import s4_supertrend as S4
from src.strategies import sma_filter as F

WARMUP_PARAMS = ("n", "sma_bars")
COST_METRIC = S4.COST_METRIC
config_params = F.config_params


def warmup(*, n: int, sma_bars: int, **_) -> int:
    return max(S4.warmup(n=n), int(sma_bars))


def target_move(df: pd.DataFrame, *, n: int, m: float, **_) -> pd.Series:
    return S4.target_move(df, n=n, m=m)


def signal(df: pd.DataFrame, *, n: int, m: float, sma_bars: int) -> pd.Series:
    """Позиция ∈ {−1, 0, 1}: S4, замаскированная фильтром SMA."""
    return F.apply(S4.signal(df, n=n, m=m), F.allowed(df, sma_bars=sma_bars))
