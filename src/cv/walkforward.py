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


def _align_up(ts: pd.Timestamp, step_months: int) -> pd.Timestamp:
    """Ближайшее начало периода (месяцы 1, 1+step, …) не раньше ts, 00:00 UTC."""
    start = pd.Timestamp(year=ts.year, month=1, day=1, tz=ts.tz)
    while start < ts:
        start = start + pd.DateOffset(months=step_months)
    return start


def folds(index: pd.DatetimeIndex, *, scheme: str, step_months: int,
          min_train_months: int, train_end: pd.Timestamp,
          usable_from: pd.Timestamp | None = None,
          rolling_train_months: int | None = None,
          allow_test: bool = False) -> list[Fold]:
    """Построить фолды по календарным периодам (шаг step_months, для квартала — 3).

    Начало данных — usable_from пары (universe.csv; данные раньше — только
    прогрев индикаторов), иначе первый бар index. Первая валидация — первое
    начало периода не раньше начала данных + min_train_months.
    expanding: train = [начало данных, val_start);
    rolling: train = [max(начало данных, val_start − rolling_train_months), val_start).
    Валидация — [val_start, val_start + step_months).

    train_end — ИСКЛЮЧИТЕЛЬНАЯ граница train этапа (2024-01-01 для train до
    2023-12-31): без allow_test ни одно окно валидации её не пересекает; с
    allow_test фолды продолжаются до конца index.

    Raises:
        ValueError: неизвестная схема, нет rolling_train_months для rolling,
            данных меньше min_train_months до train_end.
    """
    if scheme not in ("expanding", "rolling"):
        raise ValueError(f"неизвестная схема: {scheme}")
    if scheme == "rolling" and not rolling_train_months:
        raise ValueError("для rolling нужен rolling_train_months")
    data_start = index[0] if usable_from is None else max(index[0], usable_from)
    limit = train_end if not allow_test else index[-1] + pd.Timedelta(microseconds=1)
    val_start = _align_up(data_start + pd.DateOffset(months=min_train_months), step_months)
    out: list[Fold] = []
    while True:
        val_end = val_start + pd.DateOffset(months=step_months)
        if val_end > limit and not (allow_test and val_start < limit):
            break
        if scheme == "expanding":
            tr_start = data_start
        else:
            tr_start = max(data_start, val_start - pd.DateOffset(months=rolling_train_months))
        out.append(Fold(tr_start, val_start, val_start, val_end))
        val_start = val_end
    if not out:
        raise ValueError("данных меньше min_train_months до train_end")
    return out


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
    """Для каждого фолда: evaluate_grid на train → select_topk → ensemble на val.

    Сигналы считаются по всей доступной истории df (до границы, разрешённой
    load), окна train/val вырезаются после — не пересчётом с начала окна.
    Для рекурсивных стратегий (ATR Уайлдера, Supertrend) пересчёт с начала
    окна меняет позиции (зависимость от начала ряда, не look-ahead); правило
    CLAUDE.md, «Подход A».
    """
    raise NotImplementedError
