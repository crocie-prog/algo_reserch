"""Отбор вселенной гипотезы H2 без цен и доходностей (config.yaml, раздел h2.selection).

1. instruments-info (category=linear, все страницы): USDT-перпетуалы в статусе
   Trading, launchTime ≤ launch_max, без exclude_base и стейблкоинов.
2. Ликвидность — средний дневной turnover (quote) в окне [window_start,
   window_end] по дневным kline; из ответа берётся ТОЛЬКО колонка turnover,
   цены сразу отбрасываются. Нужно ровно min_days дней (полное окно).
3. Топ top_n по ликвидности.

Окно ликвидности лежит до train H2 по времени принятия решения: отбор на
2021-10-14 использует только данные до этой даты.
Ограничение: instruments-info отдаёт только торгуемые сейчас контракты
(делистинги и переименования недоступны) — смещение от выживших, вероятно
завышающее результат трендовой стратегии.

    python -m src.data.select_universe
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.config import load_config
from src.data import bybit

log = logging.getLogger("select_universe")


def list_instruments(exchange, *, retries: int = 5, backoff: float = 1.0) -> pd.DataFrame:
    """Все линейные инструменты (постранично по nextPageCursor)."""
    rows, cursor = [], None
    while True:
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        res = bybit._call(exchange, "publicGetV5MarketInstrumentsInfo", params,
                          retries=retries, backoff=backoff)["result"]
        rows += res["list"]
        cursor = res.get("nextPageCursor")
        if not cursor:
            break
    df = pd.DataFrame(rows)
    df["launch"] = pd.to_datetime(df["launchTime"].astype("int64"), unit="ms", utc=True)
    return df


def select(cfg: dict, exchange) -> tuple[pd.DataFrame, dict]:
    """(ранжированные кандидаты с turnover, воронка). Топ — первые top_n строк."""
    sc = cfg["h2"]["selection"]
    inst = list_instruments(exchange)
    funnel = {"all_linear": len(inst)}
    f = inst[(inst.quoteCoin == sc["quote"]) & (inst.contractType == "LinearPerpetual")
             & (inst.status == "Trading")]
    funnel["usdt_perp_trading"] = len(f)
    f = f[f.launch <= pd.Timestamp(sc["launch_max"], tz="UTC")]
    funnel["launch_ok"] = len(f)
    f = f[~f.baseCoin.isin(set(sc["exclude_base"]) | set(sc["stablecoins"]))]
    funnel["after_exclusions"] = len(f)
    w0 = pd.Timestamp(sc["window_start"], tz="UTC")
    w1 = pd.Timestamp(sc["window_end"], tz="UTC")
    liq = []
    for sym in f.symbol:
        turnover = bybit.fetch_klines(exchange, sym, "1d", w0, w1)["turnover"]   # только turnover
        liq.append({"symbol": sym, "days": len(turnover),
                    "avg_daily_turnover": float(turnover.mean()) if len(turnover) else float("nan")})
    liq = pd.DataFrame(liq, columns=["symbol", "days", "avg_daily_turnover"]).merge(
        f[["symbol", "baseCoin", "launch"]], on="symbol")
    ok = liq[liq.days >= int(sc["min_days"])].sort_values("avg_daily_turnover", ascending=False)
    funnel["full_window"] = len(ok)
    return ok.reset_index(drop=True), funnel


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.data.select_universe")
    p.add_argument("--config", default="config.yaml")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    ranked, funnel = select(cfg, bybit.make_exchange(cfg))
    out = cfg["paths"]["h2_candidates"]
    out.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out, index=False)
    top = ranked.head(int(cfg["h2"]["selection"]["top_n"]))["symbol"].tolist()
    log.info("воронка: %s", funnel)
    log.info("топ-%d: %s", len(top), ", ".join(top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
