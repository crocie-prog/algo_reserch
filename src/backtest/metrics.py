"""Метрики стратегии (как табл. 5 статьи), rf = 0.

Аннуализация — по числу баров в году для ТФ (365 дней).
"""
from __future__ import annotations

import pandas as pd


def trades(pos: pd.Series) -> pd.DataFrame:
    """Сделки как отрезки постоянного ненулевого знака позиции.

    Переворот закрывает одну сделку и открывает другую.
    Колонки: entry, exit, side, bars, pnl (сумма net за время удержания).
    """
    raise NotImplementedError


def metrics(bt: pd.DataFrame, *, periods_per_year: float,
            mask: pd.Series | None = None) -> dict[str, float]:
    """Метрики по результату engine.run (опционально на подвыборке mask).

    - ann_return: (equity_end / equity_start)^(P/n) − 1;
    - ann_vol: std(net) · √P;
    - sharpe: mean(net) / std(net) · √P (ddof=1);
    - mdd: максимальная просадка equity;
    - exposure: доля баров с held ≠ 0;
    - win_rate: доля сделок с pnl > 0;
    - trades_per_year; turnover_per_year (Σ|Δpos| в год);
    - n_bars, n_trades.
    """
    raise NotImplementedError
