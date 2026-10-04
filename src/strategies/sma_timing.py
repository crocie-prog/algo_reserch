"""Тайминг по SMA (контроль (б) и вторая попытка H4).

Лонг при close > SMA, шорт при close < SMA, 0 при равенстве и на прогреве;
SMA — по sma_bars рабочим барам (src.strategies.sma_filter). Параметров
сетки нет; sma_bars — из config.
"""
from __future__ import annotations

import pandas as pd

from src.strategies import sma_filter as F

WARMUP_PARAMS = ("sma_bars",)
config_params = F.config_params


def warmup(*, sma_bars: int, **_) -> int:
    return int(sma_bars)


def signal(df: pd.DataFrame, *, sma_bars: int) -> pd.Series:
    a = F.allowed(df, sma_bars=sma_bars)
    pos = a["long"].astype(float) - a["short"].astype(float)
    return pos.rename("pos")
