"""Сырые запросы к Bybit v5 через ccxt (enableRateLimit).

Используются неявные методы ccxt для эндпоинтов:
- /v5/market/kline — OHLCV и turnover (в стандартном fetch_ohlcv turnover нет);
- /v5/market/funding/history — история ставок funding;
- /v5/market/instruments-info — launchTime и статус контракта;
- /v5/market/time — серверное время (по нему определяется закрытость бара).

Поведение API (проверено 2026-10-01):
- kline: порядок по убыванию, не более 1000 за запрос, start и end включительно;
  если в окне больше limit баров — отдаются ПОСЛЕДНИЕ limit. Поэтому окно
  запроса всегда ровно limit баров: [cursor, cursor + (limit−1)·Δ].
- последний бар ответа может быть незакрытым;
- funding: порядок по убыванию, не более 200; только startTime — ошибка,
  поэтому пагинация назад по endTime.

Индекс результата — время ОТКРЫТИЯ бара, UTC, tz-aware. Сетка 1m/15m/1h/1d
выровнена от 00:00 UTC (кратна Δ от эпохи).
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Iterator

import ccxt
import numpy as np
import pandas as pd

from src.config import tf_delta

log = logging.getLogger(__name__)

INTERVALS = {"1m": "1", "15m": "15", "1h": "60", "1d": "D"}
KLINE_COLUMNS = ["open", "high", "low", "close", "volume", "turnover"]
CATEGORY = "linear"

_sleep = time.sleep   # подменяется в тестах


class DataError(RuntimeError):
    """Ответ API нарушает ожидаемые инварианты (порядок, сетка, дубли)."""


def make_exchange(cfg: dict) -> ccxt.Exchange:
    """Создать ccxt.bybit с enableRateLimit, без ключей (публичные данные).

    retry_codes — коды Bybit, при которых ExchangeError повторяется.
    """
    ex = ccxt.bybit({"enableRateLimit": bool(cfg["fetch"]["enable_rate_limit"])})
    ex.retry_codes = frozenset(int(c) for c in cfg["fetch"].get("retry_codes", []))
    return ex


_RETCODE = re.compile(r'"retCode"\s*:\s*(\d+)')


def _retryable(exchange, exc: Exception) -> bool:
    """Сетевые ошибки, лимиты и ExchangeError с кодом из exchange.retry_codes."""
    if isinstance(exc, (ccxt.NetworkError, ccxt.RateLimitExceeded)):
        return True
    if isinstance(exc, ccxt.ExchangeError):
        m = _RETCODE.search(str(exc))
        return bool(m) and int(m.group(1)) in getattr(exchange, "retry_codes", ())
    return False


def _call(exchange, method: str, params: dict, *, retries: int,
          backoff: float) -> dict:
    """Вызов неявного метода ccxt с повторами на сетевых ошибках, лимитах и
    временных серверных кодах (exchange.retry_codes).

    Ошибки запроса (BadRequest и т.п.) не повторяются.
    """
    fn = getattr(exchange, method)
    for attempt in range(retries + 1):
        try:
            return fn(params)
        except ccxt.BaseError as exc:
            if not _retryable(exchange, exc) or attempt == retries:
                raise
            wait = backoff * 2 ** attempt
            log.warning("%s %s: %s; повтор через %.1f с", method, params, exc, wait)
            _sleep(wait)
    raise AssertionError("unreachable")


def _ms(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)   # .value — всегда нс


def _ts(ms: Any) -> pd.Timestamp:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC")


def server_time(exchange, *, retries: int = 5, backoff: float = 1.0) -> pd.Timestamp:
    """Серверное время Bybit, UTC."""
    res = _call(exchange, "publicGetV5MarketTime", {}, retries=retries, backoff=backoff)
    return pd.Timestamp(int(res["result"]["timeNano"]), unit="ns", tz="UTC")


def last_closed_open(now: pd.Timestamp, tf: str, close_lag_seconds: float) -> pd.Timestamp:
    """Время открытия последнего закрытого бара: max open, для которого
    open + Δ + lag ≤ now."""
    delta = tf_delta(tf)
    return (now - pd.Timedelta(seconds=close_lag_seconds)).floor(delta) - delta


def _parse_klines(rows: list, tf: str) -> pd.DataFrame:
    """Строки API → DataFrame по возрастанию времени с проверками."""
    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS, dtype="float64",
                            index=pd.DatetimeIndex([], tz="UTC", name="open_time").as_unit("ns"))
    arr = np.asarray(rows, dtype=object)
    idx = pd.to_datetime(arr[:, 0].astype("int64"), unit="ms", utc=True)
    df = pd.DataFrame(arr[:, 1:7].astype("float64"), columns=KLINE_COLUMNS,
                      index=pd.DatetimeIndex(idx, name="open_time").as_unit("ns"))
    df = df.iloc[::-1]
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise DataError(f"kline {tf}: нарушен порядок или дубли в странице")
    step_ns = tf_delta(tf).value
    if (df.index.asi8 % step_ns != 0).any():
        raise DataError(f"kline {tf}: время открытия не на сетке Δ")
    return df


def iter_klines(exchange, symbol: str, tf: str, start: pd.Timestamp,
                end: pd.Timestamp | None = None, *, limit: int = 1000,
                close_lag_seconds: float = 5, retries: int = 5,
                backoff: float = 1.0) -> Iterator[pd.DataFrame]:
    """Свечи [start, end] постранично, по возрастанию времени.

    Окно каждого запроса — ровно limit баров, курсор сдвигается на окно
    независимо от числа строк в ответе (пустые окна до листинга и во время
    простоев биржи просто пропускаются). Незакрытые бары не отдаются никогда:
    верхняя граница — last_closed_open по серверному времени на момент старта.

    Yields:
        Непустые страницы: колонки open, high, low, close, volume (base),
        turnover (quote); индекс open_time.
    """
    delta = tf_delta(tf)
    now = server_time(exchange, retries=retries, backoff=backoff)
    upper = last_closed_open(now, tf, close_lag_seconds)
    if end is not None:
        upper = min(upper, end)
    cursor = start.ceil(delta)
    while cursor <= upper:
        req_end = min(cursor + (limit - 1) * delta, upper)
        res = _call(exchange, "publicGetV5MarketKline",
                    {"category": CATEGORY, "symbol": symbol, "interval": INTERVALS[tf],
                     "start": _ms(cursor), "end": _ms(req_end), "limit": limit},
                    retries=retries, backoff=backoff)
        page = _parse_klines(res["result"]["list"], tf)
        page = page[(page.index >= cursor) & (page.index <= req_end)]
        if len(page):
            yield page
        cursor = req_end + delta


def fetch_klines(exchange, symbol: str, tf: str, start: pd.Timestamp,
                 end: pd.Timestamp | None = None, **kw) -> pd.DataFrame:
    """Все страницы iter_klines одним DataFrame."""
    pages = list(iter_klines(exchange, symbol, tf, start, end, **kw))
    if not pages:
        return _parse_klines([], tf)
    return pd.concat(pages)


def fetch_funding(exchange, symbol: str, since: pd.Timestamp | None = None, *,
                  limit: int = 200, retries: int = 5,
                  backoff: float = 1.0) -> pd.Series:
    """История funding с метками строго позже since (None — вся история).

    Пагинация назад: endTime = серверное время, затем endTime = старейшая
    метка − 1 мс, пока страница полная и не дошли до since.

    Returns:
        Серия funding_rate; индекс — момент начисления T (UTC), по возрастанию.
        Знак: положительная ставка — лонги платят шортам.
    """
    end_ms = _ms(server_time(exchange, retries=retries, backoff=backoff))
    since_ms = _ms(since) if since is not None else None
    chunks: list[pd.Series] = []
    while True:
        params = {"category": CATEGORY, "symbol": symbol, "endTime": end_ms, "limit": limit}
        if since_ms is not None:
            params["startTime"] = since_ms + 1
        rows = _call(exchange, "publicGetV5MarketFundingHistory", params,
                     retries=retries, backoff=backoff)["result"]["list"]
        if not rows:
            break
        ts = [int(r["fundingRateTimestamp"]) for r in rows]
        chunks.append(pd.Series([float(r["fundingRate"]) for r in rows],
                                index=pd.to_datetime(ts, unit="ms", utc=True).as_unit("ns")))
        oldest = min(ts)
        if len(rows) < limit or (since_ms is not None and oldest <= since_ms + 1):
            break
        if oldest - 1 >= end_ms:
            raise DataError(f"funding {symbol}: пагинация не продвигается (endTime={end_ms})")
        end_ms = oldest - 1
    if not chunks:
        return pd.Series(dtype="float64", name="funding_rate",
                         index=pd.DatetimeIndex([], tz="UTC", name="funding_time").as_unit("ns"))
    s = pd.concat(chunks).sort_index()
    if s.index.has_duplicates:
        dup = s[s.index.duplicated(keep=False)]
        if dup.groupby(level=0).nunique().gt(1).any():
            raise DataError(f"funding {symbol}: разные ставки на одну метку")
        s = s[~s.index.duplicated()]
    if since is not None:
        s = s[s.index > since]
    s.index.name = "funding_time"
    s.name = "funding_rate"
    return s


def fetch_instrument_info(exchange, symbol: str, *, retries: int = 5,
                          backoff: float = 1.0) -> dict:
    """launch_time, status, funding_interval_minutes контракта."""
    res = _call(exchange, "publicGetV5MarketInstrumentsInfo",
                {"category": CATEGORY, "symbol": symbol}, retries=retries, backoff=backoff)
    lst = res["result"]["list"]
    if not lst:
        raise DataError(f"{symbol}: нет в instruments-info ({CATEGORY})")
    r = lst[0]
    return {"symbol": r["symbol"], "status": r["status"],
            "launch_time": _ts(r["launchTime"]),
            "funding_interval_minutes": int(r.get("fundingInterval") or 0)}
