"""Этап 6 — диагностика CPCV + PBO для сеток S1, S3, S4 на 1h train.

Без вердиктов и без test; в N для DSR не входит (журнал kind = diagnostic).

По каждой стратегии × паре и агрегату BTC+ETH (без SOL):
- CSCV/PBO, S = stage6.cscv_groups; если доля IS-баров, удалённых embargo,
  > stage6.embargo_share_alt — дополнительно S = stage6.cscv_groups_alt;
- рядом всегда P(OOS-Sharpe(n*) < 0) и медианный OOS-Sharpe n*;
- CPCV (N, k из config): Sharpe путей для (i) процедуры этапа 5 и
  (ii) лучшей по IS.
Агрегат BTC+ETH: для PBO — доходность конфигурации усредняется по двум
парам на общих барах (одна конфигурация на обе пары); для CPCV — EW путей
пар (отбор по парам, как на этапе 5).

    python -m src.cv.stage6
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, periods_per_year
from src.cv import cpcv as C
from src.cv.grid import build_grid, window_rows
from src.cv.walkforward import registered_end
from src.data.load import load, load_funding
from src.data.universe import usable_from
from src.stats import trials as T

log = logging.getLogger("stage6")


def _pbo_rows(label: dict, net: np.ndarray, pos: np.ndarray, st6: dict, ppy: float) -> list[dict]:
    h = C.embargo_bars(pos)
    out = []
    groups = [int(st6["cscv_groups"])]
    r = C.cscv(net, n_groups=groups[0], h=h, periods_per_year=ppy)
    res = [r]
    if r["embargo_share"] > st6["embargo_share_alt"]:
        res.append(C.cscv(net, n_groups=int(st6["cscv_groups_alt"]), h=h, periods_per_year=ppy))
    for x in res:
        out.append({**label, "n_groups": x["n_groups"], "h_bars": x["h"],
                    "embargo_share": x["embargo_share"], "n_configs": x["n_configs"],
                    "pbo": x["pbo"], "p_oos_loss": x["p_oos_loss"],
                    "median_oos_sharpe": x["median_oos_sharpe"],
                    "median_is_sharpe": x["median_is_sharpe"],
                    "degradation_slope": x["degradation_slope"], "n_combos": x["n_combos"],
                    "lambdas": x["lambdas"]})
    return out


def run(cfg: dict, *, strategies: list[str] | None = None,
        symbols: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    st6 = cfg["stage6"]
    tf = st6["tf"]
    ppy = periods_per_year(tf, cfg)
    end = pd.Timestamp(cfg["stage5"]["train_end_exclusive"], tz="UTC")
    strategies = strategies or st6["strategies"]
    symbols = symbols or cfg["universe"]["symbols"]
    agg_syms = st6["aggregate_symbols"]
    pbo_rows, path_rows = [], []
    for strategy in strategies:
        nets, poss, path_nets = {}, {}, {}
        for symbol in symbols:
            df = load(symbol, tf, end=registered_end(cfg), cfg=cfg)
            fund = load_funding(symbol, end=registered_end(cfg), cfg=cfg)
            start = usable_from(cfg, symbol) or df.index[0]
            gd = build_grid(df, fund, strategy=strategy, tf=tf, symbol=symbol, cfg=cfg,
                            slippage=0.0)
            rows = window_rows(gd.index, start, end)
            net = pd.DataFrame(gd.net[rows], index=gd.index[rows])
            pos = pd.DataFrame(gd.pos[rows], index=gd.index[rows])
            nets[symbol], poss[symbol] = net, pos
            lab = {"strategy": strategy, "scope": symbol}
            pbo_rows += _pbo_rows(lab, net.to_numpy(), pos.to_numpy(), st6, ppy)
            h = C.embargo_bars(pos.to_numpy())
            cp = C.cpcv(gd, df, fund, cfg, start=start, end=end, n_groups=int(st6["cpcv_groups"]),
                        k_test=int(st6["cpcv_test_groups"]), h=h, periods_per_year=ppy)
            path_nets[symbol] = cp.attrs["path_net"]
            path_rows += [{**lab, **r} for r in cp.to_dict("records")]
            log.info("%s %s: готово", strategy, symbol)
        # агрегат BTC+ETH
        avail = [s for s in agg_syms if s in nets]
        if len(avail) == len(agg_syms):
            common = nets[avail[0]].index
            for s in avail[1:]:
                common = common.intersection(nets[s].index)
            net_a = sum(nets[s].loc[common].to_numpy() for s in avail) / len(avail)
            pos_a = np.concatenate([poss[s].loc[common].to_numpy() for s in avail], axis=1)
            lab = {"strategy": strategy, "scope": "+".join(a.replace("USDT", "") for a in avail)}
            pbo_rows += _pbo_rows(lab, net_a, pos_a, st6, ppy)
            for (rule, p) in path_nets[avail[0]]:
                ew = pd.concat([path_nets[s][(rule, p)] for s in avail], axis=1, sort=True).mean(axis=1)
                ew = ew.loc[common.min():]
                sd = ew.std(ddof=1)
                path_rows.append({**lab, "rule": rule, "path": p, "n_obs": len(ew),
                                  "sharpe": float(ew.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan,
                                  "share_traded_splits": np.nan})
    pbo = pd.DataFrame(pbo_rows)
    paths = pd.DataFrame(path_rows)
    d = Path(cfg["paths"]["stage6_out"])
    d.mkdir(parents=True, exist_ok=True)
    lam = pbo[["strategy", "scope", "n_groups", "lambdas"]].explode("lambdas")
    lam.to_csv(d / "lambdas.csv", index=False)
    pbo.drop(columns="lambdas").to_csv(d / "pbo.csv", index=False)
    paths.to_csv(d / "cpcv_paths.csv", index=False)
    jr = pbo.assign(kind="diagnostic", variant=lambda x: "stage6_cscv_S" + x["n_groups"].astype(str),
                    tf=tf, symbol=pbo["scope"], fold="",
                    params=pbo.apply(lambda r: {"metric": "pbo", "h": int(r["h_bars"])}, axis=1),
                    sharpe=pbo["median_oos_sharpe"], n_obs=pbo["n_combos"],
                    note=pbo.apply(lambda r: f"pbo={r['pbo']:.3f}; p_loss={r['p_oos_loss']:.3f}", axis=1))
    T.log_rows(cfg["paths"]["trials_log"],
               jr[["kind", "variant", "strategy", "tf", "symbol", "fold", "params", "sharpe",
                   "n_obs", "note"]])
    return pbo.drop(columns="lambdas"), paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.cv.stage6")
    p.add_argument("--config", default="config.yaml")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(name)s: %(message)s")
    run(load_config(a.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
