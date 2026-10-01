"""S2: Bollinger Bands (контртренд).

mid = SMA_w, upper/lower = mid ± k·STD_w.

Машина состояний (решение на close t):
- вход long:  close_t ≤ lower_t;  вход short: close_t ≥ upper_t;
- выход long: close_t ≥ mid_t (возврат к средней линии);
  выход short: close_t ≤ mid_t;
- встречный сигнал (касание противоположной полосы) — выход;
  на том же баре вход в обратную сторону.

По сути S1 с z_exit = 0. Перед включением в МСП измерить корреляцию
доходностей с S1; если > strategies.s2.corr_with_s1_exclude (0.8) —
пометить кандидатом на исключение.
Прогрев = w баров.
"""
from __future__ import annotations

import pandas as pd

WARMUP_PARAMS = ("w",)


def warmup(*, w: int, **_) -> int:
    """Баров прогрева: w."""
    raise NotImplementedError


def signal(df: pd.DataFrame, *, w: int, k: float) -> pd.Series:
    """Позиция ∈ {−1, 0, 1} на индексе df; на прогреве 0."""
    raise NotImplementedError
