"""Векторный бэктест одной позиции на одной паре.

Сигнал считается по закрытию бара t, позиция действует с бара t+1:
    pnl_t = pos_{t−1} · ret_t − fee_t − funding_t,
где ret_t = close_t / close_{t−1} − 1 (простая доходность — аддитивна
по активам при агрегации портфеля).

Позиция ∈ [−1, 1], дробная допустима (ансамбль). Хранится полный ряд,
включая нулевые бары.
"""
from __future__ import annotations

import pandas as pd


def run(df: pd.DataFrame, pos: pd.Series, *, fee_per_side: float,
        funding: pd.Series | None = None, tf: str | None = None) -> pd.DataFrame:
    """Прогнать позицию по ценам.

    Args:
        df: свечи, индекс — время открытия бара, колонка close обязательна.
        pos: целевая позиция на закрытии бара t, тот же индекс.
        fee_per_side: доля оборота за сторону.
        funding: ставки по моментам начисления; None — без funding.
        tf: таймфрейм (нужен для привязки funding к барам).

    Returns:
        DataFrame на индексе df: pos, held (= pos.shift(1)), ret, gross,
        fee, funding, net, equity (кумулятивное произведение 1 + net).

    Raises:
        ValueError: |pos| > 1, NaN в pos, несовпадение индексов.
    """
    raise NotImplementedError
