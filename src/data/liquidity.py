"""Торгуемость пары во времени (гипотеза H2, config.yaml: h2.tradability).

Доля минут без сделок (volume == 0 в clean 1m) по календарным периодам.
Правило: в квартале валидации Q пара торгуется, только если в ПРЕДЫДУЩЕМ
периоде [Q − period, Q) доля минут без сделок ≤ max_no_trade_share; иначе
позиция 0 на весь квартал. Решение использует только данные до начала Q
(look-ahead нет). Отбор конфигураций фильтр не меняет.
"""
from __future__ import annotations

import pandas as pd

from src.data import store


def no_trade_share(cfg: dict, symbol: str, *, months: int) -> pd.Series:
    """Доля минут с нулевым объёмом по календарным кварталам (индекс — начало
    квартала, UTC), по одной годовой партиции clean 1m за раз.

    Raises:
        ValueError: months ≠ 3 (поддерживается только квартальное окно).
    """
    if months != 3:
        raise ValueError("поддерживается только квартальное окно (period_months: 3)")
    counts, zeros = {}, {}
    for f in store.partitions(cfg["paths"]["clean"], symbol, "1m"):
        d = store.read_file(f)
        key = d.index.tz_convert(None).to_period("Q").start_time.tz_localize("UTC")
        g = (d["volume"] == 0).groupby(key).agg(["size", "sum"])
        for k, v in g.iterrows():
            counts[k] = counts.get(k, 0) + int(v["size"])
            zeros[k] = zeros.get(k, 0) + int(v["sum"])
    idx = sorted(counts)
    return pd.Series([zeros[k] / counts[k] for k in idx], index=pd.DatetimeIndex(idx),
                     name="no_trade_share")


def tradable(shares: pd.Series, val_start: pd.Timestamp, *, months: int,
             max_share: float) -> bool:
    """Разрешена ли торговля в квартале с началом val_start: доля за
    предыдущий период [val_start − months, val_start) ≤ max_share.
    Нет данных за предыдущий период — торговать нельзя."""
    prev = val_start - pd.DateOffset(months=months)
    v = shares.get(prev)
    return bool(v is not None and v <= max_share)
