"""Слой data/clean: проверенные данные, которые читает load().

Правила (пропуски НЕ заполняются, бары простоя НЕ удаляются):
- 1m: raw 1m без изменений цен и объёмов, бары с open раньше начала торгов
  отбрасываются; добавляется is_downtime — плоский бар
  (open = high = low = close) с нулевым объёмом (простой биржи или минута
  без сделок).
- 15m/1h/4h/1d строятся агрегацией clean 1m (O=first, H=max, L=min, C=last,
  volume и turnover — суммы), а не берутся из родных файлов биржи: в истории
  до 2022 родные ТФ Bybit расходятся с 1m (см. отчёт качества). Родные бары
  остаются в raw для диагностики.
- Корзина старшего ТФ берётся, только если покрыта минутками целиком;
  неполные корзины (в т.ч. первая — торги BTCUSDT начались 2020-03-25 10:36)
  отбрасываются.
- Старшие ТФ: is_downtime — все минуты корзины простой; downtime_share —
  доля минут простоя.

Раскладка как в raw (1m — партиции по годам). Стык партиций (1 января
00:00 UTC) — граница корзин 15m/1h/1d, поэтому агрегация идёт по одной
партиции за раз. clean полностью производный и перезаписывается атомарно.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import tf_delta
from src.data import store

log = logging.getLogger(__name__)

BASE_TF = "1m"


def first_trade_time(cfg: dict, symbol: str) -> pd.Timestamp:
    """Начало торгов: первый сохранённый 1m-бар; без 1m — universe.first_bars."""
    files = store.partitions(cfg["paths"]["raw"], symbol, BASE_TF)
    if files:
        return store.read_file(files[0]).index[0]
    fb = cfg["universe"].get("first_bars", {}).get(symbol)
    if fb is None:
        raise ValueError(f"{symbol}: нет 1m-данных и universe.first_bars")
    return pd.Timestamp(fb, tz="UTC")


def drop_incomplete_first(df: pd.DataFrame, first_trade: pd.Timestamp) -> pd.DataFrame:
    """Удалить бары с open < first_trade."""
    return df[df.index >= first_trade]


def mark_downtime_1m(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить is_downtime: плоский 1m-бар с нулевым объёмом."""
    out = df.copy()
    flat = (out["open"] == out["high"]) & (out["high"] == out["low"]) & (out["low"] == out["close"])
    out["is_downtime"] = flat & (out["volume"] == 0)
    return out


def aggregate_1m(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Агрегат clean 1m в корзины tf (якорь 00:00 UTC); только полные корзины.

    Returns:
        open, high, low, close, volume, turnover, is_downtime, downtime_share;
        в атрибуте attrs["incomplete"] — метки отброшенных неполных корзин.
    """
    key = m1.index.floor(tf_delta(tf))
    g = m1.groupby(key)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(),
                        "volume": g["volume"].sum(), "turnover": g["turnover"].sum(),
                        "downtime_share": g["is_downtime"].mean(),
                        "n": g["close"].size()})
    out.index.name = "open_time"
    full = out["n"] == tf_delta(tf) // tf_delta(BASE_TF)
    incomplete = list(out.index[~full])
    out = out[full].drop(columns="n")
    out["is_downtime"] = out["downtime_share"] == 1.0
    out = out[["open", "high", "low", "close", "volume", "turnover",
               "is_downtime", "downtime_share"]]
    out.attrs["incomplete"] = incomplete
    return out


def build_clean_1m(cfg: dict, symbol: str) -> dict:
    """clean 1m по одной партиции raw: отброс до начала торгов + is_downtime."""
    raw_root, clean_root = cfg["paths"]["raw"], cfg["paths"]["clean"]
    files = store.partitions(raw_root, symbol, BASE_TF)
    if not files:
        return {"n_raw": 0, "n_dropped": 0, "dropped": [], "first_kept": None}
    ft = first_trade_time(cfg, symbol)
    n_raw, dropped, first_kept = 0, [], None
    for f in files:
        raw = store.read_file(f)
        out = mark_downtime_1m(drop_incomplete_first(raw, ft))
        n_raw += len(raw)
        dropped += list(raw.index[raw.index < ft])
        if first_kept is None and len(out):
            first_kept = out.index[0]
        store.write_atomic(out, store.raw_path(clean_root, symbol, BASE_TF, int(f.stem)))
    return {"n_raw": n_raw, "n_dropped": len(dropped), "dropped": dropped,
            "first_kept": first_kept}


def build_clean_from_1m(cfg: dict, symbol: str, tf: str) -> dict:
    """clean tf агрегацией clean 1m, по одной партиции 1m за раз.

    Returns:
        n_bars, n_dropped, dropped (метки неполных корзин), first_kept.
    """
    clean_root = cfg["paths"]["clean"]
    files = store.partitions(clean_root, symbol, BASE_TF)
    if not files:
        return {"n_bars": 0, "n_dropped": 0, "dropped": [], "first_kept": None}
    parts, dropped = [], []
    for f in files:
        a = aggregate_1m(store.read_file(f), tf)
        dropped += a.attrs["incomplete"]
        a.attrs = {}
        parts.append(a)
    out = pd.concat(parts)
    if out.index.has_duplicates:
        raise RuntimeError(f"{symbol} {tf}: корзина разрезана стыком партиций 1m")
    store.write_atomic(out, store.raw_path(clean_root, symbol, tf))
    if dropped:
        log.info("%s %s: отброшены неполные корзины: %d (первая %s, последняя %s)",
                 symbol, tf, len(dropped), dropped[0], dropped[-1])
    return {"n_bars": len(out), "n_dropped": len(dropped), "dropped": dropped,
            "first_kept": out.index[0] if len(out) else None}


def build_clean(cfg: dict, symbol: str, tf: str) -> dict:
    """Построить clean символа×ТФ. Для старших ТФ clean 1m должен быть построен.

    Returns:
        n_bars (баров в clean), n_dropped, dropped, first_kept.
    """
    if tf == BASE_TF:
        info = build_clean_1m(cfg, symbol)
        info["n_bars"] = info["n_raw"] - info["n_dropped"]
        return info
    return build_clean_from_1m(cfg, symbol, tf)
