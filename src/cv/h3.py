"""Прогон гипотезы H3 по предрегистрации (docs/preregistration_h3.md).

S3 и S4 на 4h, 13 пар (universe.symbols + h2.symbols), expanding, процедура
этапа 5 (ансамбль top-k, отказ при train-DSR < 0.5), фильтр торгуемости H2.
Ряд оценки — EW-OOS по парам (пары до первого квартала валидации не входят;
кварталы без торговли — с нулевой доходностью). slippage 0 — попытки
(attempt), 0.0005 — sensitivity; центр плато — diagnostic. DSR: N сквозное
(H1 + H2 + H3), уровень 2 на частоте 4h — 1h-ряды H1/H2 агрегируются в
4h-корзины суммой доходностей. Базовая линия шума — та же процедура на
перестановках суток.

    python -m src.cv.h3
В stdout — только ход прогона, без метрик.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.metrics import metrics
from src.config import load_config, periods_per_year, tf_delta
from src.cv import report
from src.cv.grid import build_grid
from src.cv.h2 import _load_pair, _sharpe, _tradable_map
from src.cv.walkforward import folds, walk_forward
from src.data.universe import usable_from
from src.stats import trials as T
from src.stats.dsr import moments, psr
from src.stats.sharpe import sharpe_se_lo

log = logging.getLogger("h3")

RULES = ("ensemble", "plateau")


def symbols_h3(cfg: dict) -> list[str]:
    return list(cfg["universe"]["symbols"]) + list(cfg["h2"]["symbols"])


def _ew(series: dict[str, pd.Series]) -> pd.Series:
    return pd.concat(series, axis=1, sort=True).mean(axis=1, skipna=True).dropna()


def to_tf(x: pd.Series, tf: str) -> pd.Series:
    """Ряд доходностей → корзины tf (сумма; корзины без наблюдений не создаются)."""
    g = x.groupby(x.index.floor(tf_delta(tf)))
    return g.sum()[g.count() > 0]


def run_variant(cfg: dict, *, strategy: str, slippage: float, symbols: list[str],
                tradable_fn=None, data_fn=None, journal: bool = True) -> dict:
    """Один прогон стратегии: OOS-ряды пар по правилам, таблицы по парам и фолдам."""
    h = cfg["h3"]
    tf, ppy = h["tf"], periods_per_year(h["tf"], cfg)
    wfc = cfg["walk_forward"]
    train_end = pd.Timestamp(cfg["stage5"]["train_end_exclusive"], tz="UTC")
    vid = f"h3_{tf}_{h['scheme']}_slip{slippage:g}"
    tradable_fn = tradable_fn or _tradable_map
    series = {r: {} for r in RULES}
    pairs, ftabs = [], []
    for sym in symbols:
        df, fund = data_fn(sym) if data_fn else _load_pair(cfg, sym, tf)
        fl = folds(df.index, scheme=h["scheme"], step_months=wfc["step_months"],
                   min_train_months=wfc["min_train_months"], train_end=train_end,
                   usable_from=usable_from(cfg, sym),
                   rolling_train_months=wfc["rolling_train_months"])
        gd = build_grid(df, fund, strategy=strategy, tf=tf, symbol=sym, cfg=cfg,
                        slippage=slippage)
        wf = walk_forward(gd, df, fund, fl, cfg, slippage=slippage, periods_per_year=ppy,
                          tradable=tradable_fn(cfg, sym, fl))
        ft = wf.folds_table
        row = {"symbol": sym, "group": "H1" if sym in cfg["universe"]["symbols"] else "H2",
               "n_folds": len(ft),
               "share_refused_dsr": float((ft["liquidity_ok"] & ~ft["trade"]).mean()),
               "share_blocked_liq": float((~ft["liquidity_ok"]).mean())}
        for rule, bt in (("ensemble", wf.bt_ensemble), ("plateau", wf.bt_plateau)):
            mask = (bt.index >= wf.oos_start) & (bt.index < wf.oos_end)
            net = bt.loc[mask, "net"].rename(sym)
            series[rule][sym] = net
            m = metrics(bt, periods_per_year=ppy, mask=mask)
            pre = "" if rule == "ensemble" else "plt_"
            row[f"{pre}oos_sharpe"] = m["sharpe"]
            row[f"{pre}oos_n_trades"] = m["n_trades"]
            if rule == "ensemble":
                for y in sorted(set(net.index.year)):
                    row[f"sharpe_{y}"] = _sharpe(net[net.index.year == y], ppy)
        pairs.append(row)
        ftabs.append(ft.assign(variant=vid))
        if journal:
            T.log_rows(cfg["paths"]["trials_log"],
                       T.train_rows(wf, gd, vid, kind="train" if slippage == 0 else "sensitivity"))
        log.info("%s %s slip=%g: готово", strategy, sym, slippage)
    return {"vid": vid, "strategy": strategy, "series": series,
            "ew": {r: _ew(series[r]) for r in RULES},
            "pairs": pd.DataFrame(pairs), "folds": pd.concat(ftabs, ignore_index=True)}


def run(cfg: dict, *, symbols: list[str] | None = None, n_perm: int | None = None,
        tradable_fn=None, data_fn=None, seed: int = 20261004) -> Path:
    """Весь прогон H3 пакетом: стратегии × slippage, шум, DSR, вердикт."""
    from src.cv.permute import permute_days

    h = cfg["h3"]
    tf, ppy = h["tf"], periods_per_year(h["tf"], cfg)
    symbols = symbols or symbols_h3(cfg)
    out = Path(cfg["paths"]["h3_out"])
    res: dict[tuple[str, float], dict] = {}
    for strategy in h["strategies"]:
        for slip in h["slippage"]:
            v = run_variant(cfg, strategy=strategy, slippage=float(slip), symbols=symbols,
                            tradable_fn=tradable_fn, data_fn=data_fn)
            d = out / v["vid"] / strategy
            d.mkdir(parents=True, exist_ok=True)
            v["pairs"].to_csv(d / "pairs.csv", index=False)
            v["folds"].to_csv(d / "folds.csv", index=False)
            rows = []
            for rule in RULES:
                v["ew"][rule].to_frame("net").to_parquet(d / f"ew_{rule}.parquet")
                kind = ("attempt" if float(slip) == 0 else "sensitivity") if rule == "ensemble" \
                    else "diagnostic"
                rows.append({"kind": kind, "variant": v["vid"], "strategy": strategy, "tf": tf,
                             "symbol": f"EW{len(symbols)}", "fold": "", "params": {"rule": rule},
                             "sharpe": _sharpe(v["ew"][rule], ppy), "n_obs": len(v["ew"][rule]),
                             "note": f"{v['vid']}/{strategy}/ew_{rule}.parquet"})
            T.log_rows(cfg["paths"]["trials_log"], pd.DataFrame(rows))
            res[(strategy, float(slip))] = v
            log.info("вариант %s %s записан", v["vid"], strategy)

    n_perm = int(n_perm if n_perm is not None else h["permutations"])
    rng = np.random.default_rng(seed)
    base = {s: (data_fn(s) if data_fn else _load_pair(cfg, s, tf))
            for s in symbols}
    noise = []
    for i in range(n_perm):
        perm = {s: (permute_days(df, rng), f) for s, (df, f) in base.items()}
        for strategy in h["strategies"]:
            v = run_variant(cfg, strategy=strategy, slippage=0.0, symbols=symbols,
                            tradable_fn=tradable_fn, data_fn=lambda s: perm[s], journal=False)
            noise.append({"perm": i, "strategy": strategy,
                          "ew_sharpe": _sharpe(v["ew"]["ensemble"], ppy),
                          "pairs_pos": int((v["pairs"]["oos_sharpe"] > 0).sum())})
        log.info("шум: перестановка %d/%d", i + 1, n_perm)
    noise = pd.DataFrame(noise, columns=["perm", "strategy", "ew_sharpe", "pairs_pos"])
    out.mkdir(parents=True, exist_ok=True)
    noise.to_csv(out / "noise.csv", index=False)
    return verdict(cfg, res, noise, symbols)


def level2_4h(cfg: dict, res: dict) -> dict:
    """Уровень 2 на частоте 4h: попытки H1 (results/stage5), H2 (results/h2)
    и H3 (attempt-ряды этого прогона)."""
    tf = cfg["h3"]["tf"]
    h1, idle = report.level2_series(cfg)
    series = {k: to_tf(v, tf) for k, v in h1.items()}
    idle = list(idle)
    h2dir = Path(cfg["paths"]["h2_out"])
    for f in sorted(h2dir.glob("h2_*_slip0/ew.parquet")):
        x = pd.read_parquet(f)["net"]
        key = f"{f.parent.name}|s4_supertrend|best_on_train"
        if x.std() > 0:
            series[key] = to_tf(x, tf)
        else:
            idle.append(key)
    for (strategy, slip), v in res.items():
        if slip != 0:
            continue
        x = v["ew"]["ensemble"]
        key = f"{v['vid']}|{strategy}|ensemble"
        if len(x) > 1 and x.std() > 0:
            series[key] = x
        else:
            idle.append(key)
    return report.level2_from_series(series, idle, cfg["selection"]["neff_share"])


def verdict(cfg: dict, res: dict, noise: pd.DataFrame, symbols: list[str]) -> Path:
    """Критерии §5 предрегистрации H3 → docs/h3_verdict.md."""
    h = cfg["h3"]
    ppy = periods_per_year(h["tf"], cfg)
    slip_hi = max(float(s) for s in h["slippage"])
    l2 = level2_4h(cfg, res)
    n_log = int((T.read_trials(cfg["paths"]["trials_log"]).kind == "attempt").sum())
    f = lambda x, nd=2: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"
    h1 = [s for s in symbols if s in cfg["universe"]["symbols"]]
    h2 = [s for s in symbols if s not in h1]
    L = ["# H3: вердикт (docs/preregistration_h3.md §5)", "",
         f"Дата: {pd.Timestamp.now(tz='UTC'):%Y-%m-%d}, git {T.git_hash()}. "
         "Train до 2023-12-31; test не открывался.", "",
         f"Уровень 2 (частота 4h): попыток {l2['n_attempts']} (в журнале attempt: {n_log}; "
         f"торговали {l2['n_traded']}, нулевых {l2['n_idle']} — один кластер), "
         f"N_eff = {l2['n_eff']}, SR₀ на 4h-бар = {l2['sr0_bar']:.3e}.", ""]
    corr = None
    ews = {}
    for strategy in h["strategies"]:
        v0, v1 = res[(strategy, 0.0)], res[(strategy, slip_hi)]
        ew = v0["ew"]["ensemble"]
        ews[strategy] = ew
        sr, se = sharpe_se_lo(ew.to_numpy(), ppy)
        t = sr / se if se and se > 0 else np.nan
        sr_b, sk, ku, n_obs = moments(ew.to_numpy())
        dsr = psr(sr_b, l2["sr0_bar"], n_obs=n_obs, skew=sk, kurt=ku) if n_obs > 2 else np.nan
        pairs = v0["pairs"]
        n_pos = int((pairs["oos_sharpe"] > 0).sum())
        sr_slip = _sharpe(v1["ew"]["ensemble"], ppy)
        base_ok = bool(sr > 0 and n_pos >= h["pairs_min_positive"] and sr_slip > 0)
        if base_ok and t >= h["t_significant"] and dsr >= h["dsr_significant"]:
            v = "Подтверждена значимо"
        elif base_ok and t >= h["t_candidate"]:
            v = "Подтверждена (кандидат)"
        else:
            v = "Не подтверждена"
        L += [f"## {strategy}", "", f"**Вердикт: {v}.**", "",
              "| Величина | Значение | Порог |", "| --- | --- | --- |",
              f"| EW-OOS Sharpe ({len(symbols)} пар) | {f(sr)} | > 0 |",
              f"| t по Lo | {f(t)} | кандидат ≥ {h['t_candidate']:g}; значимо ≥ {h['t_significant']:g} |",
              f"| DSR (N = {l2['n_attempts']}, N_eff = {l2['n_eff']}) | {f(dsr)} | значимо ≥ {h['dsr_significant']:g} |",
              f"| Пар с OOS-Sharpe > 0 | {n_pos} из {len(symbols)} | ≥ {h['pairs_min_positive']} |",
              f"| EW-OOS Sharpe при slippage {slip_hi:g} | {f(sr_slip)} | > 0 |", "",
              "Не критерии:", "",
              "| Срез | EW-OOS Sharpe | Пар > 0 |", "| --- | --- | --- |"]
        for y in sorted(set(ew.index.year)):
            col = pairs.get(f"sharpe_{y}", pd.Series(dtype=float))
            L.append(f"| {y} | {f(_sharpe(ew[ew.index.year == y], ppy))} | {int((col > 0).sum())} |")
        for name, grp in (("BTC/ETH/SOL", h1), (f"{len(h2)} пар H2", h2)):
            sub = {s: v0["series"]["ensemble"][s] for s in grp}
            pp = pairs[pairs.symbol.isin(grp)]
            L.append(f"| {name} | {f(_sharpe(_ew(sub), ppy)) if sub else '—'} | "
                     f"{int((pp.oos_sharpe > 0).sum())} из {len(grp)} |")
        L.append(f"| Центр плато (13 пар) | {f(_sharpe(v0['ew']['plateau'], ppy))} | "
                 f"{int((pairs.plt_oos_sharpe > 0).sum())} |")
        nz = noise[noise.strategy == strategy]
        if len(nz):
            L += ["", f"Шум ({len(nz)} перестановок суток): EW-OOS Sharpe среднее "
                  f"{f(nz.ew_sharpe.mean())}, 95-й перцентиль {f(nz.ew_sharpe.quantile(0.95))}; "
                  f"пар > 0: среднее {f(nz.pairs_pos.mean(), 1)}, максимум {int(nz.pairs_pos.max())}. "
                  f"Факт выше 95-го перцентиля: {'да' if sr > nz.ew_sharpe.quantile(0.95) else 'нет'}."]
        L += ["", "| Пара | Группа | Фолдов | OOS-Sharpe | Сделок OOS | Отказ по train-DSR | Закрыто ликвидностью | Центр плато |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        L += [f"| {p.symbol} | {p.group} | {p.n_folds} | {f(p.oos_sharpe)} | {int(p.oos_n_trades)} | "
              f"{f(p.share_refused_dsr)} | {f(p.share_blocked_liq)} | {f(p.plt_oos_sharpe)} |"
              for p in pairs.itertuples()]
        L.append("")
    if len(ews) == 2:
        a, b = ews.values()
        j = pd.concat([a, b], axis=1, join="inner")
        corr = float(j.corr().iloc[0, 1]) if len(j) > 2 else np.nan
        L += [f"Корреляция EW-OOS {' и '.join(ews)}: {f(corr)} (общих 4h-баров {len(j)}).", ""]
    path = Path(cfg["paths"]["h3_verdict"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.cv.h3")
    p.add_argument("--config", default="config.yaml")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(name)s: %(message)s")
    path = run(load_config(a.config))
    log.info("вердикт записан: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
