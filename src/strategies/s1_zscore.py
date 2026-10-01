"""S1: Z-score от скользящей средней (контртренд).

z_t = (close_t − SMA_w,t) / STD_w,t.

Машина состояний (решение на close t):
- вход long:  z_t < −z_entry;  вход short: z_t > z_entry;
- выход long: z_t ≥ −z_exit (возврат z к −z_exit);
  выход short: z_t ≤ z_exit;
- встречный сигнал (z за противоположным порогом входа) — выход;
  на том же баре вход в обратную сторону по условию входа.
Ограничение: 0 ≤ z_exit < z_entry.

Ловушки: whipsaw при z_exit ≈ 0; прогрев = w баров (SMA и STD).
"""
from __future__ import annotations

import pandas as pd

WARMUP_PARAMS = ("w",)


def warmup(*, w: int, **_) -> int:
    """Баров прогрева: w."""
    raise NotImplementedError


def signal(df: pd.DataFrame, *, w: int, z_entry: float, z_exit: float) -> pd.Series:
    """Позиция ∈ {−1, 0, 1} на индексе df; на прогреве 0."""
    raise NotImplementedError
