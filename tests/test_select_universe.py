"""Отбор вселенной H2: фильтры, полное окно, ранжирование по turnover (офлайн)."""
import pandas as pd

from src.data.select_universe import select
from tests.fakes import FakeBybit, make_bars


def _inst(sym, base, launch, quote="USDT", ctype="LinearPerpetual", status="Trading"):
    return {"symbol": sym, "baseCoin": base, "quoteCoin": quote, "contractType": ctype,
            "status": status, "launchTime": str(int(pd.Timestamp(launch, tz="UTC").value // 10**6)),
            "fundingInterval": 480}


def test_select_filters_and_ranks(cfg):
    inst = {s: _inst(s, b, l) for s, b, l in [
        ("AAAUSDT", "AAA", "2021-01-01"), ("BBBUSDT", "BBB", "2021-01-01"),
        ("CCCUSDT", "CCC", "2021-09-01"),                  # неполное окно
        ("BTCUSDT", "BTC", "2020-01-01"),                  # исключён
        ("USDCUSDT", "USDC", "2020-01-01"),                # стейблкоин
        ("DDDUSDT", "DDD", "2022-01-01"),                  # поздний листинг
    ]}
    inst["EEEPERP"] = _inst("EEEPERP", "EEE", "2021-01-01", quote="USDC")
    bars = {}
    for i, s in enumerate(["AAAUSDT", "BBBUSDT", "CCCUSDT", "BTCUSDT", "USDCUSDT"]):
        start = "2021-09-01" if s == "CCCUSDT" else "2021-01-01"   # свечи — с листинга
        df = make_bars(start, 400, "1D", seed=i)
        df["turnover"] = {"AAAUSDT": 1.0, "BBBUSDT": 5.0, "CCCUSDT": 99.0}.get(s, 1e6)
        bars[(s, "D")] = df
    ex = FakeBybit("2026-10-01", bars=bars, instruments=inst)
    ranked, funnel = select(cfg, ex)
    assert funnel == {"all_linear": 7, "usdt_perp_trading": 6, "launch_ok": 5,
                      "after_exclusions": 3, "full_window": 2}
    assert ranked["symbol"].tolist() == ["BBBUSDT", "AAAUSDT"]
    assert (ranked["days"] == 92).all()
