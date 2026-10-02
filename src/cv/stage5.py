"""Этап 5: пакетный прогон подхода A по предрегистрации (docs/preregistration.md).

    python -m src.cv.stage5 all            # все варианты → шум → вердикты
    python -m src.cv.stage5 run  --tf 1h --slippage 0 [--strategies ...]
    python -m src.cv.stage5 noise          # базовая линия шума (основной вариант)
    python -m src.cv.stage5 verdicts       # DSR уровня 2 и verdicts.md

Вариант = (ТФ, схема фолдов, slippage, правило выбора, окно train S6).
Сетка зависит от (пары, ТФ, slippage) и общая для схем фолдов — считается
один раз на пару. Результаты: <paths.stage5_out>/<variant>/folds.csv, scatter.csv,
oos_pairs.csv, oos/<strategy>_<symbol>_<rule>.parquet, ew/<strategy>_<rule>.parquet.
Журнал: train / attempt / sensitivity (src.stats.trials). В stdout — только ход
прогона и ошибки, без метрик доходности.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.metrics import metrics
from src.config import load_config, periods_per_year
from src.cv.grid import build_grid
from src.cv.walkforward import folds, walk_forward
from src.data.load import load, load_funding
from src.data.universe import usable_from
from src.stats import trials as T
from src.stats.sharpe import sharpe_se_lo

log = logging.getLogger("stage5")

STRATEGIES = ["s1_zscore", "s3_donchian", "s4_supertrend", "s5_pivot", "s6_vwap"]
RULES = ("ensemble", "plateau")


def out_root(cfg: dict) -> Path:
    return Path(cfg["paths"]["stage5_out"])


def variant_id(tf: str, scheme: str, slippage: float, s6_from: bool = False) -> str:
    return f"{tf}_{scheme}_slip{slippage:g}" + ("_s6from2022" if s6_from else "")


def journal_kind(tf: str, slippage: float) -> str:
    """train для основных издержек; sensitivity — для slippage > 0 (§5)."""
    return "sensitivity" if slippage > 0 else "train"


def rules_for(strategy: str) -> tuple[str, ...]:
    """S5 — одна конфигурация: центр плато совпадает с ансамблем, не отдельная попытка."""
    return ("ensemble",) if strategy == "s5_pivot" else RULES


def _pair_summary(wf, rule: str, ppy: float, min_trades: int) -> dict:
    bt = wf.bt_ensemble if rule == "ensemble" else wf.bt_plateau
    mask = (bt.index >= wf.oos_start) & (bt.index < wf.oos_end)
    m = metrics(bt, periods_per_year=ppy, mask=mask)
    sr, se = sharpe_se_lo(bt.loc[mask, "net"].to_numpy(), ppy)
    ft = wf.folds_table
    traded = ft[ft["trade"]]
    col = "ens_sharpe" if rule == "ensemble" else "plt_sharpe"
    rho = (ft.loc[ft["trade"], "best_train_sharpe"].rank()
           .corr(ft.loc[ft["trade"], col].rank()) if len(traded) > 2 else np.nan)
    return {**{f"oos_{k}": v for k, v in m.items()}, "oos_sharpe_lo": sr, "oos_se_lo": se,
            "oos_t_lo": sr / se if se and se > 0 else np.nan,
            "n_folds": len(ft), "share_no_trade": float(1 - ft["trade"].mean()),
            "share_val_pos_traded": float((traded[col] > 0).mean()) if len(traded) else np.nan,
            "spearman_train_val": rho,
            "insufficient": bool(m["n_trades"] < min_trades)}


class VariantSink:
    """Накопление результатов одного варианта до записи на диск."""

    def __init__(self, vid: str):
        self.vid = vid
        self.folds: list[pd.DataFrame] = []
        self.scatter: list[pd.DataFrame] = []
        self.pairs: list[dict] = []
        self.oos: dict[tuple[str, str], list[pd.Series]] = {}

    def save(self, root: Path) -> None:
        d = root / self.vid
        (d / "oos").mkdir(parents=True, exist_ok=True)
        (d / "ew").mkdir(parents=True, exist_ok=True)
        if self.folds:
            pd.concat(self.folds, ignore_index=True).to_csv(d / "folds.csv", index=False)
        if self.scatter:
            pd.concat(self.scatter, ignore_index=True).to_csv(d / "scatter.csv", index=False)
        pd.DataFrame(self.pairs).to_csv(d / "oos_pairs.csv", index=False)
        for (strategy, rule), series in self.oos.items():
            for s in series:
                s.to_frame("net").to_parquet(d / "oos" / f"{strategy}_{s.name}_{rule}.parquet")
            ew = pd.concat(series, axis=1, sort=True).mean(axis=1, skipna=True).dropna()
            ew.to_frame("net").to_parquet(d / "ew" / f"{strategy}_{rule}.parquet")


def run_batch(cfg: dict, *, tfs: list[str], slippages: list[float], schemes: list[str],
              strategies: list[str] | None = None, s6_variant: bool = True,
              symbols: list[str] | None = None) -> list[str]:
    """Прогнать варианты пакетом. Возвращает список id вариантов."""
    st5, wfc = cfg["stage5"], cfg["walk_forward"]
    strategies = strategies or STRATEGIES
    symbols = symbols or cfg["universe"]["symbols"]
    root = out_root(cfg)
    train_end = pd.Timestamp(st5["train_end_exclusive"], tz="UTC")
    s6_from = pd.Timestamp(st5["s6_train_from"], tz="UTC")
    done = []
    for tf in tfs:
        ppy = periods_per_year(tf, cfg)
        for slip in slippages:
            sinks: dict[str, VariantSink] = {}
            for symbol in symbols:
                df = load(symbol, tf, cfg=cfg)
                fund = load_funding(symbol, cfg=cfg)
                uf = usable_from(cfg, symbol)
                for strategy in strategies:
                    t0 = time.time()
                    gd = build_grid(df, fund, strategy=strategy, tf=tf, symbol=symbol,
                                    cfg=cfg, slippage=slip)
                    variants = [(sc, None, False) for sc in schemes]
                    if s6_variant and strategy == "s6_vwap":
                        variants += [(sc, s6_from, True) for sc in schemes]
                    for scheme, train_from, is_s6 in variants:
                        vid = variant_id(tf, scheme, slip, is_s6)
                        sink = sinks.setdefault(vid, VariantSink(vid))
                        fl = folds(df.index, scheme=scheme, step_months=wfc["step_months"],
                                   min_train_months=wfc["min_train_months"], train_end=train_end,
                                   usable_from=uf, rolling_train_months=wfc["rolling_train_months"])
                        wf = walk_forward(gd, df, fund, fl, cfg, slippage=slip,
                                          periods_per_year=ppy, train_from=train_from)
                        sink.folds.append(wf.folds_table.assign(variant=vid))
                        sink.scatter.append(wf.scatter.assign(variant=vid))
                        for rule in rules_for(strategy):
                            bt = wf.bt_ensemble if rule == "ensemble" else wf.bt_plateau
                            m = (bt.index >= wf.oos_start) & (bt.index < wf.oos_end)
                            sink.oos.setdefault((strategy, rule), []).append(
                                bt.loc[m, "net"].rename(symbol))
                            sink.pairs.append({"variant": vid, "strategy": strategy,
                                               "symbol": symbol, "rule": rule,
                                               **_pair_summary(wf, rule, ppy, st5["oos_min_trades"])})
                        T.log_rows(cfg["paths"]["trials_log"],
                                   T.train_rows(wf, gd, vid, kind=journal_kind(tf, slip)))
                    log.info("%s %s %s slip=%g: готово за %.0f с", tf, symbol, strategy, slip,
                             time.time() - t0)
            for vid, sink in sinks.items():
                sink.save(root)
                _log_level2(cfg, sink, tf, slip)
                done.append(vid)
                log.info("вариант %s записан", vid)
    return done


def _log_level2(cfg: dict, sink: VariantSink, tf: str, slip: float) -> None:
    """Записи уровня 2: attempt (1h, slippage 0) или sensitivity (1h, slippage > 0)."""
    if tf != cfg["stage5"]["tfs"][0]:
        return
    kind = "attempt" if slip == 0 else "sensitivity"
    ppy = periods_per_year(tf, cfg)
    rows = []
    for (strategy, rule), series in sink.oos.items():
        ew = pd.concat(series, axis=1, sort=True).mean(axis=1, skipna=True).dropna()
        sd = ew.std(ddof=1)
        rows.append({"kind": kind, "variant": sink.vid, "strategy": strategy, "tf": tf,
                     "symbol": "EW", "fold": "", "params": {"rule": rule},
                     "sharpe": float(ew.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan,
                     "n_obs": len(ew), "note": f"ew/{strategy}_{rule}.parquet"})
    T.log_rows(cfg["paths"]["trials_log"], pd.DataFrame(rows))


def noise_baseline(cfg: dict, *, n_perm: int | None = None, strategies: list[str] | None = None,
                   symbols: list[str] | None = None, seed: int = 20261002) -> pd.DataFrame:
    """Доля фолдов без торговли на перестановочных данных (основной вариант;
    диагностика, не попытка, в журнал не пишется)."""
    from src.cv.permute import permute_days

    st5, wfc, prim = cfg["stage5"], cfg["walk_forward"], cfg["stage5"]["primary"]
    tf = st5["tfs"][0]
    ppy = periods_per_year(tf, cfg)
    n_perm = n_perm or int(st5["permutations"])
    train_end = pd.Timestamp(st5["train_end_exclusive"], tz="UTC")
    rng = np.random.default_rng(seed)
    rows = []
    for symbol in symbols or cfg["universe"]["symbols"]:
        df0 = load(symbol, tf, cfg=cfg)
        fund = load_funding(symbol, cfg=cfg)
        fl = folds(df0.index, scheme=prim["scheme"], step_months=wfc["step_months"],
                   min_train_months=wfc["min_train_months"], train_end=train_end,
                   usable_from=usable_from(cfg, symbol),
                   rolling_train_months=wfc["rolling_train_months"])
        for i in range(n_perm):
            df = permute_days(df0, rng)
            for strategy in strategies or STRATEGIES:
                gd = build_grid(df, fund, strategy=strategy, tf=tf, symbol=symbol, cfg=cfg,
                                slippage=float(prim["slippage"]))
                from src.cv.walkforward import select_fold
                share = float(np.mean([not select_fold(gd, f, cfg, periods_per_year=ppy).trade
                                       for f in fl]))
                rows.append({"symbol": symbol, "strategy": strategy, "perm": i,
                             "share_no_trade": share})
            log.info("шум %s: перестановка %d/%d", symbol, i + 1, n_perm)
    out = pd.DataFrame(rows)
    d = out_root(cfg) / "noise"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "noise_perm.csv", index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.cv.stage5")
    p.add_argument("cmd", choices=["all", "run", "noise", "verdicts"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--tf", nargs="+")
    p.add_argument("--slippage", nargs="+", type=float)
    p.add_argument("--scheme", nargs="+")
    p.add_argument("--strategies", nargs="+")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    st5 = cfg["stage5"]
    from src.cv import report
    if a.cmd in ("all", "run"):
        run_batch(cfg, tfs=a.tf or st5["tfs"], slippages=a.slippage or st5["slippage"],
                  schemes=a.scheme or st5["schemes"], strategies=a.strategies)
    if a.cmd in ("all", "noise"):
        noise_baseline(cfg, strategies=a.strategies)
    if a.cmd in ("all", "verdicts"):
        report.verdicts(cfg)
        log.info("вердикты записаны: %s", cfg["paths"]["stage5_verdicts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
