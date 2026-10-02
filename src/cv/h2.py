"""Прогон гипотезы H2 по предрегистрации (docs/preregistration_h2.md).

S4 на 10 новых парах (config h2.symbols), 1h, expanding, правило «лучшая на
train» без отказа по train-DSR, фильтр торгуемости (доля минут без сделок в
предыдущем квартале ≤ 10%). Ряд оценки — EW-OOS по парам (пары до своего
первого квартала валидации не входят; закрытые кварталы — с нулевой
доходностью). slippage 0 — попытка (журнал attempt), 0.0005 — sensitivity.
DSR — с N сквозным: попытки H1 (results/stage5) + попытка H2.
Базовая линия шума — та же процедура на перестановках суток (диагностика).

    python -m src.cv.h2
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
from src.config import load_config, periods_per_year
from src.cv import report
from src.cv.grid import build_grid
from src.cv.walkforward import folds, walk_forward
from src.data.liquidity import no_trade_share, tradable
from src.data.load import load, load_funding
from src.data.universe import usable_from
from src.stats import trials as T
from src.stats.dsr import moments, psr
from src.stats.sharpe import sharpe_se_lo

log = logging.getLogger("h2")


def _sharpe(x: pd.Series, ppy: float) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ppy)) if len(x) > 1 and sd > 0 else np.nan


def _tradable_map(cfg: dict, symbol: str, fl) -> dict:
    t = cfg["h2"]["tradability"]
    sh = no_trade_share(cfg, symbol, months=int(t["period_months"]))
    return {f.val_start: tradable(sh, f.val_start, months=int(t["period_months"]),
                                  max_share=float(t["max_no_trade_share"])) for f in fl}


def run_variant(cfg: dict, *, slippage: float, symbols: list[str], tradable_fn=None,
                data_fn=None, journal: bool = True) -> dict:
    """Один прогон: ряды OOS пар, EW-OOS, таблицы. data_fn(symbol) → (df, funding)
    для перестановок; tradable_fn(cfg, symbol, folds) → карта торгуемости."""
    r = cfg["h2"]["run"]
    tf, ppy = r["tf"], periods_per_year(r["tf"], cfg)
    wfc = cfg["walk_forward"]
    train_end = pd.Timestamp(cfg["stage5"]["train_end_exclusive"], tz="UTC")
    vid = f"h2_{tf}_{r['scheme']}_best_slip{slippage:g}"
    tradable_fn = tradable_fn or _tradable_map
    series, pairs, ftabs = {}, [], []
    for sym in symbols:
        df, fund = data_fn(sym) if data_fn else (load(sym, tf, cfg=cfg), load_funding(sym, cfg=cfg))
        fl = folds(df.index, scheme=r["scheme"], step_months=wfc["step_months"],
                   min_train_months=wfc["min_train_months"], train_end=train_end,
                   usable_from=usable_from(cfg, sym),
                   rolling_train_months=wfc["rolling_train_months"])
        gd = build_grid(df, fund, strategy=r["strategy"], tf=tf, symbol=sym, cfg=cfg,
                        slippage=slippage)
        wf = walk_forward(gd, df, fund, fl, cfg, slippage=slippage, periods_per_year=ppy,
                          tradable=tradable_fn(cfg, sym, fl), best_requires_dsr=False)
        bt = wf.bt_best
        mask = (bt.index >= wf.oos_start) & (bt.index < wf.oos_end)
        net = bt.loc[mask, "net"].rename(sym)
        series[sym] = net
        m = metrics(bt, periods_per_year=ppy, mask=mask)
        ft = wf.folds_table
        row = {"symbol": sym, "n_folds": len(ft), "oos_sharpe": m["sharpe"],
               "oos_n_trades": m["n_trades"], "blocked_share": float(1 - ft["liquidity_ok"].mean())}
        for y in sorted(set(net.index.year)):
            row[f"sharpe_{y}"] = _sharpe(net[net.index.year == y], ppy)
        pairs.append(row)
        ftabs.append(ft.assign(variant=vid))
        if journal:
            T.log_rows(cfg["paths"]["trials_log"],
                       T.train_rows(wf, gd, vid, kind="train" if slippage == 0 else "sensitivity"))
        log.info("%s slip=%g: готово", sym, slippage)
    ew = pd.concat(series, axis=1, sort=True).mean(axis=1, skipna=True).dropna()
    return {"vid": vid, "series": series, "ew": ew, "pairs": pd.DataFrame(pairs),
            "folds": pd.concat(ftabs, ignore_index=True)}


def run(cfg: dict, *, symbols: list[str] | None = None, n_perm: int | None = None,
        tradable_fn=None, data_fn=None, seed: int = 20261003) -> Path:
    """Весь прогон H2 пакетом: варианты slippage, шум, DSR, вердикт."""
    from src.cv.permute import permute_days

    r = cfg["h2"]["run"]
    tf, ppy = r["tf"], periods_per_year(r["tf"], cfg)
    symbols = symbols or cfg["h2"]["symbols"]
    out = Path(cfg["paths"]["h2_out"])
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for slip in r["slippage"]:
        v = run_variant(cfg, slippage=float(slip), symbols=symbols, tradable_fn=tradable_fn,
                        data_fn=data_fn)
        d = out / v["vid"]
        d.mkdir(parents=True, exist_ok=True)
        v["pairs"].to_csv(d / "pairs.csv", index=False)
        v["folds"].to_csv(d / "folds.csv", index=False)
        v["ew"].to_frame("net").to_parquet(d / "ew.parquet")
        kind = "attempt" if float(slip) == 0 else "sensitivity"
        T.log_rows(cfg["paths"]["trials_log"], pd.DataFrame([{
            "kind": kind, "variant": v["vid"], "strategy": r["strategy"], "tf": tf,
            "symbol": f"EW{len(symbols)}", "fold": "", "params": {"rule": "best_on_train"},
            "sharpe": _sharpe(v["ew"], ppy), "n_obs": len(v["ew"]),
            "note": f"{v['vid']}/ew.parquet"}]))
        res[float(slip)] = v
        log.info("вариант %s записан", v["vid"])

    # базовая линия шума (slippage 0, та же процедура, карта торгуемости — по реальным данным)
    n_perm = int(n_perm if n_perm is not None else r["permutations"])
    rng = np.random.default_rng(seed)
    base = {s: (data_fn(s) if data_fn else (load(s, tf, cfg=cfg), load_funding(s, cfg=cfg)))
            for s in symbols}
    noise = []
    for i in range(n_perm):
        perm = {s: (permute_days(df, rng), f) for s, (df, f) in base.items()}
        v = run_variant(cfg, slippage=0.0, symbols=symbols, tradable_fn=tradable_fn,
                        data_fn=lambda s: perm[s], journal=False)
        noise.append({"perm": i, "ew_sharpe": _sharpe(v["ew"], ppy),
                      "pairs_pos": int((v["pairs"]["oos_sharpe"] > 0).sum())})
        log.info("шум: перестановка %d/%d", i + 1, n_perm)
    noise = pd.DataFrame(noise)
    noise.to_csv(out / "noise.csv", index=False)
    return verdict(cfg, res, noise, symbols)


def verdict(cfg: dict, res: dict, noise: pd.DataFrame, symbols: list[str]) -> Path:
    """Критерии §5 предрегистрации H2 → docs/h2_verdict.md."""
    r = cfg["h2"]["run"]
    ppy = periods_per_year(r["tf"], cfg)
    v0, v1 = res[0.0], res[sorted(res)[-1]]
    ew = v0["ew"]
    sr, se = sharpe_se_lo(ew.to_numpy(), ppy)
    t = sr / se if se and se > 0 else np.nan
    h1_series, h1_idle = report.level2_series(cfg)
    all_series = {**h1_series}
    idle = list(h1_idle)
    key = f"{v0['vid']}|{r['strategy']}|best_on_train"
    if ew.std() > 0:
        all_series[key] = ew
    else:
        idle.append(key)
    l2 = report.level2_from_series(all_series, idle, cfg["selection"]["neff_share"])
    sr_b, sk, ku, n_obs = moments(ew.to_numpy())
    dsr = psr(sr_b, l2["sr0_bar"], n_obs=n_obs, skew=sk, kurt=ku)
    pairs = v0["pairs"]
    n_pos = int((pairs["oos_sharpe"] > 0).sum())
    sr_slip = _sharpe(v1["ew"], ppy)
    years = sorted(set(ew.index.year))
    by_year = {y: _sharpe(ew[ew.index.year == y], ppy) for y in years}
    pos_year = {y: int((pairs.get(f"sharpe_{y}", pd.Series(dtype=float)) > 0).sum()) for y in years}
    base_ok = sr > 0 and n_pos >= r["pairs_min_positive"] and sr_slip > 0
    if base_ok and t >= r["t_significant"] and dsr >= r["dsr_significant"]:
        v = "Подтверждена значимо"
    elif base_ok and t >= r["t_candidate"]:
        v = "Подтверждена (кандидат)"
    else:
        v = "Не подтверждена"
    f = lambda x, nd=2: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"
    L = ["# H2: вердикт (docs/preregistration_h2.md §5)", "",
         f"Дата: {pd.Timestamp.now(tz='UTC'):%Y-%m-%d}, git {T.git_hash()}. "
         "Train до 2023-12-31; test не открывался.", "",
         f"**Вердикт: {v}.**", "",
         "## Критерии", "",
         "| Величина | Значение | Порог |", "| --- | --- | --- |",
         f"| EW-OOS Sharpe ({len(symbols)} пар) | {f(sr)} | > 0 |",
         f"| t по Lo | {f(t)} | кандидат ≥ {r['t_candidate']:g}; значимо ≥ {r['t_significant']:g} |",
         f"| DSR (N = {l2['n_attempts']}, N_eff = {l2['n_eff']}) | {f(dsr)} | значимо ≥ {r['dsr_significant']:g} |",
         f"| Пар с OOS-Sharpe > 0 | {n_pos} из {len(symbols)} | ≥ {r['pairs_min_positive']} |",
         f"| EW-OOS Sharpe при slippage {sorted(res)[-1]:g} | {f(sr_slip)} | > 0 |", "",
         f"Уровень 2: попыток {l2['n_attempts']} (торговали {l2['n_traded']}, нулевых "
         f"{l2['n_idle']} — один кластер), N_eff = {l2['n_eff']}, SR₀ на бар = {l2['sr0_bar']:.3e}.", "",
         "## По годам (не критерий)", "", "| Год | EW-OOS Sharpe | Пар > 0 |", "| --- | --- | --- |"]
    L += [f"| {y} | {f(by_year[y])} | {pos_year[y]} |" for y in years]
    L += ["", "## По парам (slippage 0)", "",
          "| Пара | Фолдов | OOS-Sharpe | Сделок OOS | Кварталов закрыто фильтром торгуемости |",
          "| --- | --- | --- | --- | --- |"]
    L += [f"| {p.symbol} | {p.n_folds} | {f(p.oos_sharpe)} | {int(p.oos_n_trades)} | {f(p.blocked_share)} |"
          for p in pairs.itertuples()]
    if len(noise):
        L += ["", "## Базовая линия шума (перестановки суток, диагностика)", "",
              f"Перестановок: {len(noise)}; EW-OOS Sharpe: среднее {f(noise.ew_sharpe.mean())}, "
              f"95-й перцентиль {f(noise.ew_sharpe.quantile(0.95))}; пар > 0: среднее "
              f"{f(noise.pairs_pos.mean(), 1)}, максимум {int(noise.pairs_pos.max())}. "
              f"Факт выше 95-го перцентиля шума: {'да' if sr > noise.ew_sharpe.quantile(0.95) else 'нет'}."]
    path = Path(cfg["paths"]["h2_verdict"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.cv.h2")
    p.add_argument("--config", default="config.yaml")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(name)s: %(message)s")
    path = run(load_config(a.config))
    log.info("вердикт записан: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
