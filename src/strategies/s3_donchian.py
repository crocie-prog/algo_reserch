"""S3: Donchian Channel (тренд).

HH_t = max(high_{t−w}, …, high_{t−1}), LL_t = min(low_{t−w}, …, low_{t−1}) —
канал по барам ДО t (сдвиг на 1), иначе пробоя нет или он смотрит в будущее.

Машина состояний (решение на close t):
- вход long:  close_t > HH_t;  вход short: close_t < LL_t;
- выход long: close_t < LL_t (противоположная граница канала);
  выход short: close_t > HH_t;
- встречный сигнал совпадает с выходом → переворот на том же баре.

Следствие: после первого входа стратегия всегда в рынке (стоп-и-переворот),
экспозиция ≈ 100%, во флэте — частые перевороты.
Прогрев = w + 1 баров.
"""
from __future__ import annotations

import pandas as pd

WARMUP_PARAMS = ("w",)


def warmup(*, w: int, **_) -> int:
    """Баров прогрева: w + 1."""
    raise NotImplementedError


def signal(df: pd.DataFrame, *, w: int) -> pd.Series:
    """Позиция ∈ {−1, 0, 1} на индексе df; на прогреве 0."""
    raise NotImplementedError
