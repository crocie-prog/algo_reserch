"""Общие индикаторы и каркас машины состояний для S1–S6.

Правило машины состояний (едино для всех стратегий), на закрытии бара t:
1. В позиции: если выполнено собственное условие выхода стратегии ИЛИ
   встречный сигнал (условие входа в противоположную сторону) — позиция
   закрывается.
2. Если позиция пуста (в том числе только что закрыта на этом баре) —
   проверяется условие входа; при выполнении — вход.
Следствие: встречный сигнал на одном баре даёт выход и вход в обратную
сторону (переворот, два оборота комиссии).

Позиция — решение на close t; сдвиг на бар делает только движок.
На прогреве позиция = 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: pd.Series, w: int) -> pd.Series:
    """Простая скользящая средняя, min_periods = w."""
    raise NotImplementedError


def rolling_std(x: pd.Series, w: int) -> pd.Series:
    """Скользящее стандартное отклонение, ddof=1, min_periods = w."""
    raise NotImplementedError


def true_range(df: pd.DataFrame) -> pd.Series:
    """TR_t = max(H−L, |H−C_{t−1}|, |L−C_{t−1}|)."""
    raise NotImplementedError


def atr_wilder(df: pd.DataFrame, n: int) -> pd.Series:
    """ATR по Уайлдеру: первое значение — среднее TR за n баров,
    далее ATR_t = ((n−1)·ATR_{t−1} + TR_t) / n. До бара n — NaN."""
    raise NotImplementedError


def run_state_machine(entry_long: np.ndarray, entry_short: np.ndarray,
                      exit_long: np.ndarray, exit_short: np.ndarray,
                      warmup: int) -> np.ndarray:
    """Пройти бары по правилу из docstring модуля.

    Встречный сигнал для лонга — entry_short, для шорта — entry_long;
    передавать его в exit_* не нужно. NaN в условиях трактуется как False.

    Returns:
        Массив позиций ∈ {−1, 0, 1}; [:warmup] = 0.
    """
    raise NotImplementedError
