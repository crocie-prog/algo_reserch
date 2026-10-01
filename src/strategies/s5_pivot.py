"""S5: Pivot Points S1/R1 (диапазон).

Уровни дня D (UTC, граница 00:00) — только от ЗАКРЫТОГО дня D−1:
  P = (H + L + C)/3,  S1 = 2P − H,  R1 = 2P − L  (H, L, C дня D−1).
День D−1 агрегируется из баров рабочего ТФ; неполный день D−1 (пропуски
баров) — уровни дня D = NaN, входов нет.
Уровни действуют на всех барах дня D — классическое место смещения баров:
бар 23:00 дня D использует уровни D, бар 00:00 дня D+1 — уже уровни D+1.

Машина состояний (решение на close t):
- вход long:  close_t < S1_D;  вход short: close_t > R1_D;
- выход long: close_t ≥ P_D (возврат к P); выход short: close_t ≤ P_D;
- встречный сигнал (close за противоположным уровнем) — выход и переворот.
Позиция переносится через 00:00 UTC; после смены дня выход проверяется
по уровням нового дня. Закрытие в конце дня — отдельная гипотеза.

Прогрев: до первого бара дня, для которого определён полный предыдущий день.
"""
from __future__ import annotations

import pandas as pd

WARMUP_PARAMS = ()


def warmup(df: pd.DataFrame | None = None, **_) -> int:
    """Баров прогрева: индекс первого бара второго полного дня UTC."""
    raise NotImplementedError


def daily_levels(df: pd.DataFrame) -> pd.DataFrame:
    """P, S1, R1 для каждого бара df по закрытому предыдущему дню UTC."""
    raise NotImplementedError


def signal(df: pd.DataFrame) -> pd.Series:
    """Позиция ∈ {−1, 0, 1} на индексе df; на прогреве 0."""
    raise NotImplementedError
