"""Фейковая биржа, эмулирующая поведение Bybit v5 (без сети).

Эмулируется:
- kline: [start, end] включительно, порядок по убыванию, при переполнении —
  последние limit баров окна; без end — limit баров от start вперёд;
  бары с open > now отсутствуют, бар, содержащий now, — незакрытый (отдаётся);
- funding: только startTime — BadRequest; порядок по убыванию; последние limit;
- time, instruments-info;
- fail_first: первые N вызовов kline падают с NetworkError.
"""
from __future__ import annotations

import ccxt
import numpy as np
import pandas as pd


def make_bars(start: str, periods: int, freq: str, seed: int = 0) -> pd.DataFrame:
    """Синтетические бары на сетке freq: случайное блуждание, согласованный OHLC."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC", name="open_time").as_unit("ns")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, periods)))
    open_ = np.r_[100.0, close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, periods))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, periods))
    vol = rng.uniform(1, 10, periods)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": vol, "turnover": vol * close}, index=idx)


def _ms(ts) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


class FakeBybit:
    def __init__(self, now, bars=None, funding=None, instruments=None, fail_first=0):
        self.now = pd.Timestamp(now)
        self.bars = bars or {}            # (symbol, interval) -> DataFrame
        self.funding = funding or {}      # symbol -> Series
        self.instruments = instruments or {}
        self.fail_first = fail_first
        self.calls = []

    def publicGetV5MarketTime(self, params):
        return {"result": {"timeNano": str(self.now.value),
                           "timeSecond": str(self.now.value // 10**9)}}

    def publicGetV5MarketKline(self, params):
        self.calls.append(("kline", dict(params)))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise ccxt.NetworkError("fake network error")
        df = self.bars.get((params["symbol"], params["interval"]))
        lim = int(params.get("limit", 200))
        if df is None:
            return {"result": {"list": []}}
        ms = df.index.as_unit("ns").asi8 // 1_000_000
        visible = df[ms <= _ms(self.now)]
        vms = visible.index.as_unit("ns").asi8 // 1_000_000
        if "end" in params:
            sel = visible[(vms >= int(params["start"])) & (vms <= int(params["end"]))].iloc[-lim:]
        else:
            sel = visible[vms >= int(params["start"])].iloc[:lim]
        rows = [[str(int(t.value // 1_000_000))] + [repr(float(v)) for v in r]
                for t, r in zip(sel.index, sel[["open", "high", "low", "close",
                                                "volume", "turnover"]].values)]
        return {"result": {"list": rows[::-1]}}

    def publicGetV5MarketFundingHistory(self, params):
        self.calls.append(("funding", dict(params)))
        if "startTime" in params and "endTime" not in params:
            raise ccxt.BadRequest("params error: Time Is Invalid")
        s = self.funding.get(params["symbol"], pd.Series(dtype=float))
        ms = s.index.as_unit("ns").asi8 // 1_000_000 if len(s) else np.array([], dtype="int64")
        mask = ms <= int(params.get("endTime", _ms(self.now)))
        if "startTime" in params:
            mask &= ms >= int(params["startTime"])
        sel = s[mask].iloc[-int(params.get("limit", 200)):]
        rows = [{"symbol": params["symbol"], "fundingRate": repr(float(v)),
                 "fundingRateTimestamp": str(int(t.value // 1_000_000))}
                for t, v in zip(sel.index, sel.values)]
        return {"result": {"list": rows[::-1]}}

    def publicGetV5MarketInstrumentsInfo(self, params):
        info = self.instruments.get(params["symbol"])
        return {"result": {"list": [info] if info else []}}
