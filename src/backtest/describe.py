"""Описательный прогон стратегий на реальных данных (этап 3).

Только частотные величины позиций: сделок в год, экспозиция, средняя
длительность сделки, доля переворотов — и ex-ante cost_to_target.
Доходность, Sharpe и equity НЕ считаются (до этапа 5).

cost_to_target = издержки круга / медиана цели сделки в долях цены,
издержки круга = 2 · (fee_per_side + slippage_per_side) из config (0.002);
цель сделки даёт стратегия (target_move), медиана — по барам окна (от
usable_from). Для S1: (z_entry − z_exit) · STD_w / close. Правило отсечения
по этому показателю решается на этапе 5, до walk-forward.

Имя показателя задаёт стратегия (COST_METRIC = (имя медианы, имя отношения));
по умолчанию ("median_target", "cost_to_target"). У S3 это cost_to_width —
издержки к ширине канала, с cost_to_target в одной таблице не смешивать.

Это не подбор параметров: в журнал попыток DSR ничего не пишется.

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
             symbols: list[str] | None = None,
             from_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Таблица описательных величин: строка на пару × набор параметров.

    Параметры, не входящие в сетку (config_params стратегии, например якорь
    и прогрев сессии S6), подставляются из config. from_date — начало окна
    не раньше этой даты (например, 2022-01-01 для S6); окно = max(usable_from,
    from_date). Для сессионной S6 добавляются доли сделок, перенесённых
    через 00:00 UTC, и среди них — закрытых в первые min_session_bars баров.
    """
    mod = importlib.import_module(f"src.strategies.{strategy}")
    rows = []
    for symbol in symbols or cfg["universe"]["symbols"]:
        df = load(symbol, tf, cfg=cfg)
        uf = usable_from(cfg, symbol)
        if from_date is not None:
            fd = pd.Timestamp(from_date, tz="UTC")
            uf = fd if uf is None else max(uf, fd)
        c = cfg["costs"]
        round_trip = 2 * (float(c["fee_per_side"]) + float(c.get("slippage_per_side", 0.0)))
        for params in params_list:
            if hasattr(mod, "config_params"):
                params = {**mod.config_params(tf, cfg), **params}
            pos = mod.signal(df, **params)
            tgt = mod.target_move(df, **params) if hasattr(mod, "target_move") else None
            if uf is not None:
                pos = pos[pos.index >= uf]
                tgt = tgt[tgt.index >= uf] if tgt is not None else None
            med = float(tgt.median()) if tgt is not None else float("nan")
            med_name, ratio_name = getattr(mod, "COST_METRIC", ("median_target", "cost_to_target"))
            st = position_stats(pos, periods_per_year=periods_per_year(tf, cfg))
            hours = tf_delta(tf) / pd.Timedelta(hours=1)
            extra = {}
            if hasattr(mod, "session_carry_stats") and params.get("anchor") == "session":
                extra = mod.session_carry_stats(df.loc[pos.index], pos,
                                                min_session_bars=params["min_session_bars"])
            rows.append({"symbol": symbol, "tf": tf, "params": json.dumps(params),
                         "from": pos.index[0], "to": pos.index[-1],
                         "trades_per_year": st["trades_per_year"],
                         "exposure": st["exposure"],
                         "mean_duration_h": st["mean_duration_bars"] * hours,
                         "reversal_share": st["reversal_share"],
                         "n_trades": st["n_trades"],
                         med_name: med,
                         ratio_name: round_trip / med if med > 0 else float("nan"),
                         **extra})
    return pd.DataFrame(rows)


def position_agreement(strategy_a: str, params_a: dict, strategy_b: str, params_b: dict,
                       *, tf: str, cfg: dict, symbols: list[str] | None = None) -> pd.DataFrame:
    """Сходство ПОЗИЦИЙ двух стратегий (не доходностей) по каждой паре.

    Окно — train от usable_from; сигналы считаются по всей загруженной
    истории. corr_pos — корреляция Пирсона рядов позиций; share_equal —
    доля баров с одинаковой позицией.
    """
    ma = importlib.import_module(f"src.strategies.{strategy_a}")
    mb = importlib.import_module(f"src.strategies.{strategy_b}")
    rows = []
    for symbol in symbols or cfg["universe"]["symbols"]:
        df = load(symbol, tf, cfg=cfg)
        a, b = ma.signal(df, **params_a), mb.signal(df, **params_b)
        uf = usable_from(cfg, symbol)
        if uf is not None:
            a, b = a[a.index >= uf], b[b.index >= uf]
        rows.append({"symbol": symbol, "tf": tf,
                     "a": f"{strategy_a} {json.dumps(params_a)}",
                     "b": f"{strategy_b} {json.dumps(params_b)}",
                     "corr_pos": float(a.corr(b)),
                     "share_equal": float((a == b).mean())})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.backtest.describe")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--strategy", required=True)
    p.add_argument("--tf", default="1h")
    p.add_argument("--params", nargs="+", required=True, help="JSON-словари параметров")
    p.add_argument("--out", help="CSV для сохранения")
    p.add_argument("--from", dest="from_date", help="начало окна не раньше даты (YYYY-MM-DD)")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    tab = describe(a.strategy, [json.loads(s) for s in a.params], tf=a.tf, cfg=cfg,
                   from_date=a.from_date)
    if a.out:
        tab.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
