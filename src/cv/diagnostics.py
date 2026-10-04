"""Диагностика (не попытки): валовая и чистая доходность всей сетки на train.

Для каждой стратегии × пары на 1h train (от usable_from до train_end этапа)
по ВСЕЙ сетке, без отбора и фолдов:
- Sharpe до издержек: комиссия 0, slippage 0, funding не учитывается;
- Sharpe после издержек: комиссия из config, funding по факту, slippage 0.
Сводка: медиана, доля положительных, максимум — для обоих.
Журнал: kind = diagnostic (в N для DSR не входит).

    python -m src.cv.diagnostics gross
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys

import pandas as pd

from src.config import load_config, periods_per_year
from src.cv.grid import build_grid, window_stats
from src.cv.walkforward import registered_end
from src.data.load import load, load_funding
from src.data.universe import usable_from
from src.stats import trials as T

log = logging.getLogger("diagnostics")


def gross_vs_net(cfg: dict, *, tf: str = "1h", strategies: list[str] | None = None,
                 symbols: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(по конфигурациям, сводка по стратегии × паре)."""
    from src.cv.stage5 import STRATEGIES, out_root

    ppy = periods_per_year(tf, cfg)
    end = pd.Timestamp(cfg["stage5"]["train_end_exclusive"], tz="UTC")
    gross_cfg = copy.deepcopy(cfg)
    gross_cfg["costs"]["fee_per_side"] = 0.0
    gross_cfg["costs"]["include_funding"] = False
    rows = []
    for symbol in symbols or cfg["universe"]["symbols"]:
        df = load(symbol, tf, end=registered_end(cfg), cfg=cfg)
        fund = load_funding(symbol, end=registered_end(cfg), cfg=cfg)
        start = usable_from(cfg, symbol) or df.index[0]
        for strategy in strategies or STRATEGIES:
            out = {}
            for label, c in (("gross", gross_cfg), ("net", cfg)):
                gd = build_grid(df, fund, strategy=strategy, tf=tf, symbol=symbol, cfg=c,
                                slippage=0.0)
                out[label] = window_stats(gd, start, end, periods_per_year=ppy)["sharpe"]
            for j, p in enumerate(gd.grid_params):
                rows.append({"strategy": strategy, "symbol": symbol, "tf": tf, "params": p,
                             "sharpe_gross": float(out["gross"].iloc[j]),
                             "sharpe_net": float(out["net"].iloc[j])})
            log.info("%s %s: готово", symbol, strategy)
    per = pd.DataFrame(rows)
    g = per.groupby(["strategy", "symbol"])
    summ = pd.DataFrame({
        "n_configs": g.size(),
        "gross_median": g["sharpe_gross"].median(),
        "gross_share_pos": g["sharpe_gross"].apply(lambda s: float((s > 0).mean())),
        "gross_max": g["sharpe_gross"].max(),
        "net_median": g["sharpe_net"].median(),
        "net_share_pos": g["sharpe_net"].apply(lambda s: float((s > 0).mean())),
        "net_max": g["sharpe_net"].max(),
    }).reset_index()
    d = out_root(cfg) / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    per.to_csv(d / "gross_vs_net_configs.csv", index=False)
    summ.to_csv(d / "gross_vs_net_summary.csv", index=False)
    jr = pd.concat([
        per.assign(kind="diagnostic", variant=f"{tf}_train_full_{lab}", fold="",
                   sharpe=per[f"sharpe_{lab}"], n_obs=0, note=f"без отбора, {lab}")
        for lab in ("gross", "net")], ignore_index=True)
    T.log_rows(cfg["paths"]["trials_log"],
               jr[["kind", "variant", "strategy", "tf", "symbol", "fold", "params", "sharpe",
                   "n_obs", "note"]])
    return per, summ


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.cv.diagnostics")
    p.add_argument("cmd", choices=["gross"])
    p.add_argument("--config", default="config.yaml")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(name)s: %(message)s")
    gross_vs_net(load_config(a.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
