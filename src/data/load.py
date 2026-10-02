"""Единая точка чтения данных для исследований (слой clean).

Защита тестового периода: по умолчанию данные обрезаются на train_end.
Запрос с end > train_end без allow_test=True вызывает PeriodAccessError.

Границы:
- start — включительно по времени открытия бара;
- end — включительно; если у end нет времени (00:00), это ДАТА и
  включается весь день: open < end + 1 день. train_end = 2023-12-31 →
  последний бар 1h — 2023-12-31 23:00.
- funding: метка T включается, если T ≤ исключительной верхней границы
  баров, то есть начисление 2024-01-01 00:00 входит в train: оно относится
  к последнему бару train (бар, заканчивающийся в T).
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import load_config
from src.data import store

log = logging.getLogger(__name__)


class PeriodAccessError(RuntimeError):
    """Попытка прочитать тестовый период без явного разрешения."""


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _bounds(cfg: dict, start, end, allow_test: bool) -> tuple[pd.Timestamp | None, pd.Timestamp]:
    """(нижняя включительная, верхняя ИСКЛЮЧИТЕЛЬНАЯ) граница по open."""
    train_excl = _utc(cfg["periods"]["train_end"]) + pd.Timedelta(days=1)
    test_excl = _utc(cfg["periods"]["test_end"]) + pd.Timedelta(days=1)
    if end is None:
        upper = test_excl if allow_test else train_excl
    else:
        e = _utc(end)
        upper = e + pd.Timedelta(days=1) if e == e.normalize() else e + pd.Timedelta(1, "ns")
    if upper > train_excl and not allow_test:
        raise PeriodAccessError(
            f"запрос до {upper} заходит в тестовый период (train_end = "
            f"{cfg['periods']['train_end']}); нужен allow_test=True")
    if allow_test and upper > train_excl:
        log.warning("доступ к тестовому периоду: данные до %s", upper)
    return (_utc(start) if start is not None else None), upper


def load(symbol: str, tf: str, start: str | pd.Timestamp | None = None,
         end: str | pd.Timestamp | None = None, *, allow_test: bool = False,
         cfg: dict | None = None) -> pd.DataFrame:
    """Свечи символа×ТФ из clean в [start, end].

    Индекс — время открытия бара, UTC, tz-aware. Колонки: open, high, low,
    close, volume, turnover, is_downtime; у 15m/1h/1d ещё downtime_share
    (старшие ТФ — агрегат 1m, только полные корзины). end=None → train_end
    (или test_end при allow_test). Пропуски не заполняются, бары простоя
    не удаляются. Данные до universe.warmup_only_until — только прогрев.

    Raises:
        PeriodAccessError: end > train_end и allow_test=False.
        FileNotFoundError: нет данных в clean.
    """
    cfg = cfg or load_config()
    lo, hi = _bounds(cfg, start, end, allow_test)
    df = store.read(cfg["paths"]["clean"], symbol, tf, lo, hi)
    if df is None:
        raise FileNotFoundError(f"нет данных clean: {symbol} {tf}")
    return df[df.index < hi]


def load_funding(symbol: str, start: str | pd.Timestamp | None = None,
                 end: str | pd.Timestamp | None = None, *, allow_test: bool = False,
                 cfg: dict | None = None) -> pd.Series:
    """Ставки funding: индекс — момент начисления T, UTC; T ≤ исключительной
    верхней границы баров (см. docstring модуля). Та же защита test."""
    cfg = cfg or load_config()
    lo, hi = _bounds(cfg, start, end, allow_test)
    df = store.read(cfg["paths"]["raw"], symbol, "funding", lo, hi)
    if df is None:
        raise FileNotFoundError(f"нет funding: {symbol}")
    return df["funding_rate"]
