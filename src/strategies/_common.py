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
На прогреве и там, где индикатор не определён (NaN), действий нет.

Бары простоя (колонка is_downtime в clean; в старших ТФ ещё downtime_share):
- состояние заморожено: новых позиций нет, открытая позиция держится и не
  закрывается — биржа не торгует, выход физически невозможен; первое
  решение после простоя — на первом рабочем баре;
- бары простоя не входят в скользящие окна средних и σ: окно отсчитывается
  по рабочим барам (working_rolling).
Данные до universe.warmup_only_until (BTCUSDT до 2021-01-01) — только прогрев
индикаторов, не обучение и не оценка (позиции обнуляет движок, active_from).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def downtime_mask(df: pd.DataFrame) -> np.ndarray:
    """is_downtime как bool-массив; без колонки — простоя нет."""
    if "is_downtime" in df:
        return df["is_downtime"].to_numpy(dtype=bool)
    return np.zeros(len(df), dtype=bool)


def working_rolling(x: pd.Series, w: int, is_downtime: np.ndarray | None,
                    how: str) -> pd.Series:
    """Скользящая статистика по рабочим барам.

    Бары простоя удаляются, окно w считается по оставшимся (min_periods = w),
    результат возвращается на исходный индекс; на барах простоя — NaN.

    Args:
        how: "mean" или "std" (ddof=1).
    """
    if w < 1:
        raise ValueError("w ≥ 1")
    work = x if is_downtime is None or not is_downtime.any() else x[~is_downtime]
    r = work.rolling(w, min_periods=w)
    if how == "mean":
        out = r.mean()
    elif how == "std":
        out = r.std(ddof=1)
    else:
        raise ValueError(f"неизвестная статистика: {how}")
    return out.reindex(x.index)


def sma(x: pd.Series, w: int, is_downtime: np.ndarray | None = None) -> pd.Series:
    """Простая скользящая средняя по рабочим барам, min_periods = w."""
    return working_rolling(x, w, is_downtime, "mean")


def rolling_std(x: pd.Series, w: int, is_downtime: np.ndarray | None = None) -> pd.Series:
    """Скользящее стандартное отклонение по рабочим барам, ddof=1, min_periods = w."""
    return working_rolling(x, w, is_downtime, "std")


def true_range(df: pd.DataFrame) -> pd.Series:
    """TR_t = max(H−L, |H−C_{t−1}|, |L−C_{t−1}|). Реализуется с S4."""
    raise NotImplementedError


def atr_wilder(df: pd.DataFrame, n: int) -> pd.Series:
    """ATR по Уайлдеру: первое значение — среднее TR за n баров,
    далее ATR_t = ((n−1)·ATR_{t−1} + TR_t) / n. До бара n — NaN. Реализуется с S4."""
    raise NotImplementedError


def run_state_machine(entry_long: np.ndarray, entry_short: np.ndarray,
                      exit_long: np.ndarray, exit_short: np.ndarray,
                      frozen: np.ndarray | None = None, warmup: int = 0) -> np.ndarray:
    """Пройти бары по правилу из docstring модуля.

    Встречный сигнал для лонга — entry_short, для шорта — entry_long;
    передавать его в exit_* не нужно. Условия — bool-массивы (NaN индикатора
    вызывающий код превращает в False). На барах frozen (простой) состояние
    не меняется. На [:warmup] позиция 0.

    Returns:
        Массив позиций ∈ {−1., 0., 1.}.
    """
    el, es = np.asarray(entry_long, bool), np.asarray(entry_short, bool)
    xl, xs = np.asarray(exit_long, bool), np.asarray(exit_short, bool)
    n = len(el)
    if not (len(es) == len(xl) == len(xs) == n):
        raise ValueError("массивы условий разной длины")
    fr = np.zeros(n, bool) if frozen is None else np.asarray(frozen, bool)
    out = np.zeros(n)
    state = 0.0
    for t in range(max(warmup, 0), n):
        if fr[t]:
            out[t] = state
            continue
        if state > 0 and (xl[t] or es[t]):
            state = 0.0
        elif state < 0 and (xs[t] or el[t]):
            state = 0.0
        if state == 0.0:
            if el[t] and not es[t]:
                state = 1.0
            elif es[t] and not el[t]:
                state = -1.0
        out[t] = state
    return out
