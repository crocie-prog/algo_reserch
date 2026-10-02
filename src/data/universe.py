"""data/meta/universe.csv: символ, статус, первый и последний бар по ТФ.

fixed_on (universe.fixed_on) — дата фиксации СОСТАВА вселенной (листинг
последней пары), а не обрезка истории: BTC и ETH используют всю доступную
историю. Делистингованные контракты API может не отдавать — survivorship bias.

warmup_only_until (universe.warmup_only_until) — данные до этой даты только для
прогрева индикаторов (BTCUSDT 2020 — неликвидный период). usable_from —
начало периода обучения и оценки: max(first_trade, warmup_only_until).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import bybit, clean, store

FIXED_ON_NOTE = ("fixed_on — дата фиксации состава вселенной, не обрезка истории; "
                 "каждая пара использует всю доступную историю. Данные до usable_from "
                 "(warmup_only_until) — только прогрев индикаторов, не обучение и не оценка")


def usable_from(cfg: dict, symbol: str) -> pd.Timestamp | None:
    """Начало обучения и оценки для пары по конфигу: universe.warmup_only_until
    (данные раньше — только прогрев). None — ограничения нет (данные clean
    и так начинаются с начала торгов)."""
    wu = cfg["universe"].get("warmup_only_until", {}).get(symbol)
    return pd.Timestamp(wu, tz="UTC") if wu else None


def build_universe(cfg: dict, exchange=None, *, symbols: list[str] | None = None,
                   path=None) -> pd.DataFrame:
    """Собрать таблицу по слою clean (и instruments-info, если передан exchange)
    и записать в path (по умолчанию paths.meta; для H2 — paths.meta_h2)."""
    rows = []
    for symbol in symbols or cfg["universe"]["symbols"]:
        row: dict = {"symbol": symbol}
        if exchange is not None:
            info = bybit.fetch_instrument_info(exchange, symbol,
                                               retries=int(cfg["fetch"]["retries"]),
                                               backoff=float(cfg["fetch"]["backoff_seconds"]))
            row.update(status=info["status"], launch_time=info["launch_time"],
                       funding_interval_minutes=info["funding_interval_minutes"])
        try:
            row["first_trade"] = clean.first_trade_time(cfg, symbol)
        except ValueError:
            row["first_trade"] = None
        for tf in cfg["timeframes"]["download"]:
            files = store.partitions(cfg["paths"]["clean"], symbol, tf)
            if files:
                first = store.read_file(files[0]).index[0]
                last = store.last_timestamp(cfg["paths"]["clean"], symbol, tf)
            else:
                first = last = None
            row[f"first_bar_{tf}"] = first
            row[f"last_bar_{tf}"] = last
        f = store.read(cfg["paths"]["raw"], symbol, "funding")
        row["first_funding"] = f.index[0] if f is not None and len(f) else None
        row["last_funding"] = f.index[-1] if f is not None and len(f) else None
        wu = cfg["universe"].get("warmup_only_until", {}).get(symbol)
        row["warmup_only_until"] = pd.Timestamp(wu, tz="UTC") if wu else None
        starts = [t for t in (row["first_trade"], row["warmup_only_until"]) if t is not None]
        row["usable_from"] = max(starts) if starts else None
        row["fixed_on"] = cfg["universe"]["fixed_on"]
        row["note"] = FIXED_ON_NOTE
        rows.append(row)
    df = pd.DataFrame(rows)
    path = Path(path) if path is not None else cfg["paths"]["meta"]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
