"""Описательный прогон стратегий на реальных данных (этап 3).

Только частотные величины позиций: сделок в год, экспозиция, средняя
длительность сделки, доля переворотов. Доходность, Sharpe и equity НЕ
считаются (до этапа 5). Это не подбор параметров: в журнал попыток DSR
ничего не пишется.

Окно — train (load по умолчанию) начиная с usable_from пары; сигнал
считается по всей загруженной истории (прогрев индикаторов), позиции
до usable_from не учитываются.

CLI: python -m src.backtest.describe --strategy s1_zscore --tf 1h
     --params '{"w": 24, "z_entry": 2, "z_exit": 0.5}' [...]
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

import pandas as pd

from src.backtest.metrics import position_stats
from src.config import load_config, periods_per_year, tf_delta
from src.data.load import load
from src.data.universe import usable_from


def describe(strategy: str, params_list: list[dict], *, tf: str, cfg: dict,
             symbols: list[str] | None = None) -> pd.DataFrame:
    """Таблица описательных величин: строка на пару × набор параметров."""
    mod = importlib.import_module(f"src.strategies.{strategy}")
    rows = []
    for symbol in symbols or cfg["universe"]["symbols"]:
        df = load(symbol, tf, cfg=cfg)
        uf = usable_from(cfg, symbol)
        for params in params_list:
            pos = mod.signal(df, **params)
            if uf is not None:
                pos = pos[pos.index >= uf]
            st = position_stats(pos, periods_per_year=periods_per_year(tf, cfg))
            hours = tf_delta(tf) / pd.Timedelta(hours=1)
            rows.append({"symbol": symbol, "tf": tf, "params": json.dumps(params),
                         "from": pos.index[0], "to": pos.index[-1],
                         "trades_per_year": st["trades_per_year"],
                         "exposure": st["exposure"],
                         "mean_duration_h": st["mean_duration_bars"] * hours,
                         "reversal_share": st["reversal_share"],
                         "n_trades": st["n_trades"]})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.backtest.describe")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--strategy", required=True)
    p.add_argument("--tf", default="1h")
    p.add_argument("--params", nargs="+", required=True, help="JSON-словари параметров")
    p.add_argument("--out", help="CSV для сохранения")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    tab = describe(a.strategy, [json.loads(s) for s in a.params], tf=a.tf, cfg=cfg)
    if a.out:
        tab.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
