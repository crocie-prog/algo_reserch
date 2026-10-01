"""Трёхуровневая агрегация: пара → портфель стратегии → МСП.

Основной вариант МСП — неттинг: позиции всех стратегий по одной паре
складываются с весами в одну нетто-позицию, комиссия и funding — с неё
(один счёт). Вариант «отдельные счета» (комиссия с каждой стратегии
отдельно) считается для отчёта.

rf = 0: свободный капитал ничего не приносит; рычага нет, Σ|w·pos| ≤ 1.
"""
from __future__ import annotations

import pandas as pd


def strategy_portfolio(pos_by_pair: dict[str, pd.Series], w: pd.Series) -> dict[str, pd.Series]:
    """Позиции портфеля одной стратегии по парам: w_pair · pos_pair."""
    raise NotImplementedError


def net_positions(pos_by_strategy: dict[str, dict[str, pd.Series]],
                  w: pd.Series) -> dict[str, pd.Series]:
    """Нетто-позиция по каждой паре: Σ_s w_s · pos_{s,pair}."""
    raise NotImplementedError


def build_levels(data: dict[str, pd.DataFrame], pos: dict[str, dict[str, pd.Series]],
                 cfg: dict, *, tf: str, netting: bool = True) -> dict[str, pd.DataFrame]:
    """Доходности уровней pair, strategy, msp (net, с издержками).

    Args:
        data: свечи по парам.
        pos: позиции {стратегия: {пара: ряд}}.
        netting: True — один счёт, False — отдельные счета.
    """
    raise NotImplementedError
