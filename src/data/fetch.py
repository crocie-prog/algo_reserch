"""CLI инкрементальной загрузки.

    python -m src.data.fetch --config config.yaml [--symbols ...] [--tf ...]
                             [--no-funding] [--no-quality] [--refetch-last N]

Первый запуск — с launchTime контракта, повторный — только новое.
Перед докачкой последний сохранённый бар запрашивается заново и сверяется:
если биржа его пересмотрела — ошибка (RevisionError). --refetch-last N
осознанно перекачивает и перезаписывает последние N баров с записью
в журнал paths.refetch_log.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, tf_delta
from src.data import bybit, store

log = logging.getLogger("src.data.fetch")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Аргументы CLI."""
    p = argparse.ArgumentParser(prog="python -m src.data.fetch",
                                description="Инкрементальная загрузка Bybit (линейные перпетуалы).")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--symbols", nargs="+", help="по умолчанию — universe.symbols")
    p.add_argument("--tf", nargs="+", help="по умолчанию — timeframes.download")
    p.add_argument("--no-funding", action="store_true", help="не качать funding")
    p.add_argument("--no-quality", action="store_true", help="не строить clean и отчёт")
    p.add_argument("--refetch-last", type=int, default=0, metavar="N",
                   help="перекачать и перезаписать последние N баров (пересмотр биржей)")
    return p.parse_args(argv)


def _fetch_kw(cfg: dict) -> dict:
    f = cfg["fetch"]
    return {"limit": int(f["kline_limit"]), "close_lag_seconds": float(f["close_lag_seconds"]),
            "retries": int(f["retries"]), "backoff": float(f["backoff_seconds"])}


def _log_refetch(cfg: dict, row: dict) -> None:
    path = Path(cfg["paths"]["refetch_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _verify_last_bar(ex, cfg: dict, symbol: str, tf: str, last: pd.Timestamp) -> None:
    """Сверить сохранённый последний бар с текущим ответом API."""
    root = cfg["paths"]["raw"]
    saved = store.read(root, symbol, tf, last, last)
    fresh = bybit.fetch_klines(ex, symbol, tf, last, last, **_fetch_kw(cfg))
    if fresh.empty:
        raise store.RevisionError(
            f"{symbol} {tf}: последний сохранённый бар {last} больше не отдаётся API; "
            f"см. --refetch-last")
    a = saved[bybit.KLINE_COLUMNS].to_numpy(dtype="float64")
    b = fresh[bybit.KLINE_COLUMNS].to_numpy(dtype="float64")
    if not np.array_equal(a, b, equal_nan=True):
        raise store.RevisionError(
            f"{symbol} {tf}: биржа пересмотрела бар {last}: было {a.tolist()}, стало "
            f"{b.tolist()}; перезапуск с --refetch-last N перезапишет последние N баров")


def update_klines(ex, cfg: dict, symbol: str, tf: str, *, refetch_last: int = 0) -> int:
    """Докачать один символ×ТФ. Возвращает число новых баров."""
    root = cfg["paths"]["raw"]
    delta = tf_delta(tf)
    last = store.last_timestamp(root, symbol, tf)
    overwrite_from = None
    if last is None:
        info = bybit.fetch_instrument_info(ex, symbol, retries=int(cfg["fetch"]["retries"]),
                                           backoff=float(cfg["fetch"]["backoff_seconds"]))
        start = info["launch_time"].floor(delta)
        log.info("%s %s: первая загрузка с %s", symbol, tf, start)
    elif refetch_last > 0:
        start = last - (refetch_last - 1) * delta
        overwrite_from = start
        log.warning("%s %s: --refetch-last %d, перезапись с %s", symbol, tf, refetch_last, start)
    else:
        _verify_last_bar(ex, cfg, symbol, tf, last)
        start = last + delta

    old_tail = store.read(root, symbol, tf, overwrite_from) if overwrite_from is not None else None
    chunk = int(cfg["fetch"]["chunk_bars"])
    buf: list[pd.DataFrame] = []
    n_buf = added = 0
    fetched_tail: list[pd.DataFrame] = []

    def flush() -> None:
        nonlocal buf, n_buf, added
        if buf:
            part = pd.concat(buf)
            added += store.write(root, symbol, tf, part, overwrite_from=overwrite_from)
            log.info("%s %s: записано до %s (+%d)", symbol, tf, part.index[-1], len(part))
        buf, n_buf = [], 0

    for page in bybit.iter_klines(ex, symbol, tf, start, **_fetch_kw(cfg)):
        if overwrite_from is not None:
            fetched_tail.append(page)
        buf.append(page)
        n_buf += len(page)
        if n_buf >= chunk:
            flush()
    flush()

    if overwrite_from is not None:
        new_tail = pd.concat(fetched_tail) if fetched_tail else bybit._parse_klines([], tf)
        common = old_tail.index.intersection(new_tail.index) if old_tail is not None else []
        n_changed = 0
        if len(common):
            a = old_tail.loc[common, bybit.KLINE_COLUMNS].to_numpy(dtype="float64")
            b = new_tail.loc[common, bybit.KLINE_COLUMNS].to_numpy(dtype="float64")
            n_changed = int((a != b).any(axis=1).sum())
        _log_refetch(cfg, {"time_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                           "symbol": symbol, "tf": tf, "from": start.isoformat(),
                           "n_requested": refetch_last, "n_compared": len(common),
                           "n_changed": n_changed})
        log.warning("%s %s: перезаписано, изменено баров: %d", symbol, tf, n_changed)
    return added


def update_funding(ex, cfg: dict, symbol: str) -> int:
    """Докачать историю funding. Возвращает число новых записей."""
    root = cfg["paths"]["raw"]
    last = store.last_timestamp(root, symbol, "funding")
    s = bybit.fetch_funding(ex, symbol, since=last, limit=int(cfg["fetch"]["funding_limit"]),
                            retries=int(cfg["fetch"]["retries"]),
                            backoff=float(cfg["fetch"]["backoff_seconds"]))
    n = store.write(root, symbol, "funding", s.to_frame())
    log.info("%s funding: +%d (последнее %s)", symbol, n, s.index[-1] if len(s) else last)
    return n


def _setup_logging(cfg: dict) -> None:
    path = Path(cfg["paths"]["fetch_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI. Код возврата 0 — успех, 1 — ошибка пересмотра данных."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    _setup_logging(cfg)
    symbols = args.symbols or cfg["universe"]["symbols"]
    tfs = args.tf or cfg["timeframes"]["download"]
    ex = bybit.make_exchange(cfg)
    try:
        for symbol in symbols:
            for tf in tfs:
                n = update_klines(ex, cfg, symbol, tf, refetch_last=args.refetch_last)
                log.info("%s %s: новых баров %d", symbol, tf, n)
            if not args.no_funding:
                update_funding(ex, cfg, symbol)
    except store.RevisionError as exc:
        log.error("%s", exc)
        return 1
    if not args.no_quality:
        from src.data import quality, universe
        quality.run_all(cfg, symbols=symbols, tfs=tfs)
        universe.build_universe(cfg, ex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
