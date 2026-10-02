"""Детекторы look-ahead для стратегий (этап 3 и далее).

- truncation_check: позиция, посчитанная только по df[:k], должна совпадать
  с позицией по полному df на первых k барах. Стратегия, использующая
  будущее (shift(−1), центрированное окно, нормировка по всей выборке),
  даёт расхождение.
- shift_check: Sharpe позиции, сдвинутой на −1 бар (заглядывание на бар),
  сравнивается с исходным. Честная стратегия от сдвига «чудесно» не
  улучшается; утечка проявляется как Sharpe(исходной) ≈ Sharpe(сдвинутой)
  при абсурдно высоких значениях.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.backtest import engine
from src.backtest.metrics import metrics


class LookAheadError(AssertionError):
    """Позиция на баре t зависит от данных после t."""


def truncation_check(signal_fn: Callable[..., pd.Series], df: pd.DataFrame,
                     params: dict | None = None, cuts: Iterable[int] | None = None,
                     atol: float = 0.0) -> None:
    """Проверить, что pos[:k] по df[:k] совпадает с pos[:k] по полному df.

    Args:
        cuts: длины префиксов; по умолчанию 10 точек равномерно по ряду.

    Raises:
        LookAheadError: с первым расходящимся баром.
    """
    params = params or {}
    full = signal_fn(df, **params)
    n = len(df)
    cuts = list(cuts) if cuts is not None else list(np.linspace(n // 10, n - 1, 10, dtype=int))
    for k in cuts:
        part = signal_fn(df.iloc[:k], **params)
        a, b = full.iloc[:k].to_numpy(dtype="float64"), part.to_numpy(dtype="float64")
        bad = ~np.isclose(a, b, rtol=0, atol=atol, equal_nan=True)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise LookAheadError(
                f"префикс {k}: позиция на {df.index[i]} по полному ряду {a[i]}, "
                f"по префиксу {b[i]} — сигнал зависит от будущих баров")


def shift_check(df: pd.DataFrame, pos: pd.Series, *, periods_per_year: float,
                fee_per_side: float = 0.0) -> dict[str, float]:
    """Sharpe исходной позиции и той же позиции со сдвигом на −1 бар.

    Returns:
        sharpe (исходная), sharpe_lead (pos.shift(−1), заглядывание на бар),
        delta = sharpe_lead − sharpe.
    """
    lead = pos.shift(-1).fillna(0.0)
    s0 = metrics(engine.run(df, pos, fee_per_side=fee_per_side),
                 periods_per_year=periods_per_year)["sharpe"]
    s1 = metrics(engine.run(df, lead, fee_per_side=fee_per_side),
                 periods_per_year=periods_per_year)["sharpe"]
    return {"sharpe": s0, "sharpe_lead": s1, "delta": s1 - s0}
