"""Журнал попыток для DSR: results/trials_log.csv (в git).

Пишется с первого прогона сетки. Регистрируется каждая оценённая
конфигурация и каждый вариант дизайна, включая неудачные и отброшенные.
Записи только добавляются, не удаляются.
"""
from __future__ import annotations

import pandas as pd


def log_trial(path: str, *, stage: str, strategy: str, tf: str, params: dict,
              window: str, sharpe: float, n_obs: int, note: str = "") -> None:
    """Дописать одну попытку (с меткой времени и git-хэшем)."""
    raise NotImplementedError


def read_trials(path: str) -> pd.DataFrame:
    """Прочитать журнал."""
    raise NotImplementedError


def n_trials(path: str, **filters) -> int:
    """Число попыток для DSR с фильтрами (стратегия, ТФ, этап)."""
    raise NotImplementedError
