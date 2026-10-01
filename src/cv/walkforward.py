"""Walk-forward подхода A: фолды, ранжирование сетки, топ-k, ансамбль, порог.

Параметры — config.yaml (walk_forward, selection). Шаг — квартал,
минимальное обучающее окно — 12 месяцев. Схемы: expanding и rolling.

Весь подбор — только на train (до periods.train_end). Тестовое окно позже
train_end фолды без allow_test=True не создают.

Каждый прогон сетки пишется в журнал попыток (src.stats.trials) — для DSR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class Fold:
    """Один фолд: обучение [train_start, train_end), валидация [val_start, val_end)."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def folds(index: pd.DatetimeIndex, *, scheme: str, step_months: int,
          min_train_months: int, train_end: pd.Timestamp,
          rolling_train_months: int | None = None,
          allow_test: bool = False) -> list[Fold]:
    """Построить фолды по календарным кварталам.

    Первый фолд начинается, когда накоплено min_train_months данных.
    expanding: train_start = начало данных; rolling: train_start =
    val_start − rolling_train_months.

    Без allow_test ни одно окно валидации не выходит за train_end
    (последнее усекается или отбрасывается).

    Raises:
        ValueError: неизвестная схема, данных меньше min_train_months.
    """
    raise NotImplementedError


def evaluate_grid(df: pd.DataFrame, strategy, grid: list[dict], mask: pd.Series,
                  *, fee_per_side: float, funding: pd.Series | None,
                  tf: str, periods_per_year: float) -> pd.DataFrame:
    """Прогнать сетку на всей истории df, метрики считать на mask.

    Индикаторы каузальные, поэтому считаются на всей доступной истории;
    состояние позиции переносится через границы окон.

    Returns:
        По строке на набор параметров: params, sharpe, n_trades,
        trades_per_year, exposure, ann_return, mdd.
    """
    raise NotImplementedError


def select_topk(scores: pd.DataFrame, *, k: int, min_train_sharpe: float,
                min_trades_per_year: float) -> list[dict]:
    """Топ-k по Sharpe среди наборов с trades_per_year ≥ min_trades_per_year.

    Если лучший Sharpe < min_train_sharpe или проходящих нет — пустой
    список: в этом окне стратегия не торгует.
    """
    raise NotImplementedError


def ensemble(df: pd.DataFrame, strategy, params_list: list[dict]) -> pd.Series:
    """Позиция ансамбля: mean_k sign(pos_k) ∈ [−1, 1]; пустой список → 0."""
    raise NotImplementedError


@dataclass
class WFResult:
    """Позиции по окнам валидации, склеенные в один ряд, и журнал отбора по фолдам."""

    pos: pd.Series
    selection: pd.DataFrame


def walk_forward(df: pd.DataFrame, strategy, grid: list[dict], fold_list: list[Fold],
                 cfg: dict, *, tf: str, funding: pd.Series | None = None,
                 log_trial: Callable | None = None) -> WFResult:
    """Для каждого фолда: evaluate_grid на train → select_topk → ensemble на val."""
    raise NotImplementedError
