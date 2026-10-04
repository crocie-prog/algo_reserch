"""Прогон гипотезы H4 по предрегистрации (docs/preregistration_h4.md).

Попытки (EW-OOS BTC/ETH/SOL, все фолды 2022–2024, expanding до 2024-12-31):
1. S4-4h + фильтр SMA_200d (src.strategies.s4_sma), протокол H3/H3b с отбором;
2. тайминг SMA_200d (src.strategies.sma_timing), без отбора.
Контроли (diagnostic): (а) S4-4h без фильтра, (в) buy & hold. Позиции
тайминга и B&H действуют только в разрешённых фильтром торгуемости
кварталах валидации, как у стратегий с отбором.

Вердикт по каждой попытке (§4): кандидат — Sharpe > 0, t по Lo ≥ 1, знак при
slippage 0.0005, торговали ≥ 2 из 3, все торговавшие > 0; значимо — плюс
t ≥ 2 и DSR ≥ 0.95 (N сквозное, H3b — отдельный кластер). Кандидат для test —
по правилу §5. Шум — доля перестановок суток, прошедших «кандидата»
(≥ weak_share → «слабо»). ΔSharpe — Ledoit–Wolf (HAC), не в критериях.

    python -m src.cv.h4
В stdout — только ход прогона, без метрик.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import engine
from src.backtest.metrics import metrics
from src.config import load_config, periods_per_year
from src.cv import report
from src.cv.grid import window_rows
from src.cv.h2 import _sharpe, _tradable_map
from src.cv.h3 import to_tf
from src.cv.h3b import _end, _ew, _load_pair, _quarters, criterion, fold_list, run_variant
from src.data.universe import usable_from
from src.stats import trials as T
from src.stats.dsr import expected_max_sharpe, moments, psr
from src.stats.lw import lw_sharpe_diff
from src.stats.sharpe import sharpe_se_lo
from src.strategies import sma_filter, sma_timing

log = logging.getLogger("h4")
SEC = "h4"


def run_fixed(cfg: dict, *, kind: str, slippage: float, symbols: list[str],
              tradable_fn=None, data_fn=None) -> dict:
    """Стратегии без отбора: kind = "timing" (SMA) или "bh" (buy & hold).
    Позиция действует в разрешённых кварталах валидации, вне них — 0."""
    h = cfg[SEC]
    tf, ppy = h["tf"], periods_per_year(h["tf"], cfg)
    tradable_fn = tradable_fn or _tradable_map
    c = cfg["costs"]
    out = {"net": {}, "long": {}, "short": {}, "pos": {}}
    pairs = []
    for sym in symbols:
        df, fund = data_fn(sym) if data_fn else _load_pair(cfg, sym, tf, SEC)
        fl = fold_list(cfg, SEC, sym, df.index)
        trad = tradable_fn(cfg, sym, fl)
        on = np.zeros(len(df), bool)
        for f in fl:
            if trad.get(f.val_start, False):
                on[window_rows(df.index, f.val_start, f.val_end)] = True
        if kind == "timing":
            base = sma_timing.signal(df, **sma_timing.config_params(tf, cfg)).to_numpy()
        else:
            base = np.ones(len(df))
        pos = pd.Series(np.where(on, base, 0.0), index=df.index)
        bt = engine.run(df, pos, fee_per_side=float(c["fee_per_side"]), slippage_per_side=slippage,
                        funding=fund if c.get("include_funding", True) else None, tf=tf,
                        active_from=usable_from(cfg, sym))
        mask = (bt.index >= fl[0].val_start) & (bt.index < fl[-1].val_end)
        w = bt.loc[mask]
        net = w["net"].rename(sym)
        out["net"][sym] = net
        out["long"][sym] = net.where(w["held"] > 0, 0.0)
        out["short"][sym] = net.where(w["held"] < 0, 0.0)
        out["pos"][sym] = w["pos"]
        m = metrics(bt, periods_per_year=ppy, mask=mask)
        pairs.append({"symbol": sym, "n_folds": len(fl), "oos_sharpe": m["sharpe"],
                      "oos_n_trades": m["n_trades"], "traded": bool(m["n_trades"] > 0),
                      "share_long": float((w["held"] > 0).mean()),
                      "share_short": float((w["held"] < 0).mean()),
                      "share_blocked_liq": float(1 - np.mean([trad.get(f.val_start, False)
                                                              for f in fl]))})
    return {"vid": f"{SEC}_{kind}_{tf}_slip{slippage:g}", "series": out, "ew": _ew(out["net"]),
            "ew_long": _ew(out["long"]), "ew_short": _ew(out["short"]),
            "pairs": pd.DataFrame(pairs)}


def assess(cfg: dict, v0: dict, v1: dict) -> dict:
    """Критерий «кандидат» §4 (без DSR): компоненты и t."""
    h = cfg[SEC]
    ppy = periods_per_year(h["tf"], cfg)
    ok, c = criterion(v0["ew"], v1["ew"], v0["pairs"], ppy=ppy, min_traded=int(h["min_traded"]))
    sr, se = sharpe_se_lo(v0["ew"].to_numpy(), ppy) if len(v0["ew"]) > 2 else (np.nan, np.nan)
    t = sr / se if se and se > 0 else np.nan
    cand = bool(ok and t >= h["t_candidate"])
    return {**c, "t": t, "candidate": cand}


def level2(cfg: dict, attempts: dict[str, pd.Series]) -> dict:
    """Уровень 2 на 4h: H1–H3 и попытки H4 на общих барах; H3b — отдельный кластер."""
    tf = cfg[SEC]["tf"]
    h1, idle = report.level2_series(cfg)
    series = {k: to_tf(v, tf) for k, v in h1.items()}
    idle = list(idle)
    globs = [(cfg["paths"]["h2_out"], "h2_*_slip0/ew.parquet"),
             (cfg["paths"]["h3_out"], "h3_*_slip0/*/ew_ensemble.parquet")]
    for root, pat in globs:
        for f in sorted(Path(root).glob(pat)):
            x = pd.read_parquet(f)["net"]
            key = f"{Path(root).name}|{f.parent.name}"
            if len(x) > 1 and x.std() > 0:
                series[key] = to_tf(x, tf) if "h2" in key else x
            else:
                idle.append(key)
    for k, x in attempts.items():
        if len(x) > 1 and x.std() > 0:
            series[f"h4|{k}"] = x
        else:
            idle.append(f"h4|{k}")
    base = report.level2_from_series(series, idle, cfg["selection"]["neff_share"])
    mat = pd.concat(series, axis=1, join="inner", sort=True)
    sr_bar = list((mat.mean() / mat.std(ddof=1)).to_numpy())
    h3b = Path(cfg["paths"]["h3b_out"]) / "target_slip0" / "ew.parquet"
    n_extra = 0
    if h3b.exists():
        x = pd.read_parquet(h3b)["net"]
        n_extra = 1
        if len(x) > 1 and x.std() > 0:
            sr_bar.append(float(x.mean() / x.std(ddof=1)))
    v = float(np.var(sr_bar, ddof=1)) if len(sr_bar) > 1 else 0.0
    n_eff = base["n_eff"] + n_extra
    return {"n_attempts": base["n_attempts"] + n_extra, "n_eff_common": base["n_eff"],
            "n_eff": n_eff, "n_common_bars": len(mat), "var_sr_bar": v,
            "sr0_bar": expected_max_sharpe(n_eff, v)}


def run(cfg: dict, *, n_perm: int | None = None, tradable_fn=None, data_fn=None,
        seed: int = 20261006) -> Path:
    """Весь прогон H4 пакетом."""
    from src.cv.permute import permute_days

    h = cfg[SEC]
    tf, ppy = h["tf"], periods_per_year(h["tf"], cfg)
    target, second = list(cfg["universe"]["symbols"]), list(cfg["h2"]["symbols"])
    out = Path(cfg["paths"]["h4_out"])
    slips = [float(s) for s in h["slippage"]]
    hi = max(slips)
    kw = dict(tradable_fn=tradable_fn, data_fn=data_fn, section=SEC)
    R = {}
    for s in slips:
        R[("s4f", s)] = run_variant(cfg, slippage=s, symbols=target, strategy=h["strategy"],
                                    journal_kind="train" if s == 0 else "sensitivity", **kw)
        R[("timing", s)] = run_fixed(cfg, kind="timing", slippage=s, symbols=target,
                                     tradable_fn=tradable_fn, data_fn=data_fn)
        log.info("попытки при slippage %g: готово", s)
    R[("s4", 0.0)] = run_variant(cfg, slippage=0.0, symbols=target, strategy=h["control"],
                                 journal_kind="diagnostic", **kw)
    R[("bh", 0.0)] = run_fixed(cfg, kind="bh", slippage=0.0, symbols=target,
                               tradable_fn=tradable_fn, data_fn=data_fn)
    sec = {"s4f": run_variant(cfg, slippage=0.0, symbols=second, strategy=h["strategy"],
                              journal_kind="diagnostic", **kw),
           "timing": run_fixed(cfg, kind="timing", slippage=0.0, symbols=second,
                               tradable_fn=tradable_fn, data_fn=data_fn),
           "bh": run_fixed(cfg, kind="bh", slippage=0.0, symbols=second,
                           tradable_fn=tradable_fn, data_fn=data_fn)}
    log.info("контроли и вторичная вселенная: готово")
    for (k, s), v in R.items():
        d = out / f"target_{k}_slip{s:g}"
        d.mkdir(parents=True, exist_ok=True)
        v["pairs"].to_csv(d / "pairs.csv", index=False)
        v["ew"].to_frame("net").to_parquet(d / "ew.parquet")
        if "folds" in v:
            v["folds"].to_csv(d / "folds.csv", index=False)
    for k, v in sec.items():
        d = out / f"secondary_{k}_slip0"
        d.mkdir(parents=True, exist_ok=True)
        v["pairs"].to_csv(d / "pairs.csv", index=False)
        v["ew"].to_frame("net").to_parquet(d / "ew.parquet")

    def row(kind, key, s, rule, ew, sym=f"EW{len(target)}"):
        return {"kind": kind, "variant": f"{SEC}_{key}_{tf}_slip{s:g}", "strategy": key, "tf": tf,
                "symbol": sym, "fold": "", "params": {"rule": rule},
                "sharpe": _sharpe(ew, ppy), "n_obs": len(ew), "note": f"target_{key}_slip{s:g}"}
    rows = [row("attempt", "s4f", 0.0, "ensemble", R[("s4f", 0.0)]["ew"]),
            row("attempt", "timing", 0.0, "fixed", R[("timing", 0.0)]["ew"]),
            row("sensitivity", "s4f", hi, "ensemble", R[("s4f", hi)]["ew"]),
            row("sensitivity", "timing", hi, "fixed", R[("timing", hi)]["ew"]),
            row("diagnostic", "s4", 0.0, "ensemble", R[("s4", 0.0)]["ew"]),
            row("diagnostic", "bh", 0.0, "fixed", R[("bh", 0.0)]["ew"]),
            row("diagnostic", "s4f", 0.0, "plateau", R[("s4f", 0.0)]["ew_plateau"]),
            row("diagnostic", "s4f", 0.0, "ensemble", sec["s4f"]["ew"],
                sym=f"EW{len(second)}_secondary")]
    T.log_rows(cfg["paths"]["trials_log"], pd.DataFrame(rows))
    log.info("журнал записан")

    n_perm = int(n_perm if n_perm is not None else h["permutations"])
    rng = np.random.default_rng(seed)
    base = {s: (data_fn(s) if data_fn else _load_pair(cfg, s, tf, SEC)) for s in target}
    noise = []
    for i in range(n_perm):
        perm = {s: (permute_days(df, rng), f) for s, (df, f) in base.items()}
        pk = dict(tradable_fn=tradable_fn, data_fn=lambda s: perm[s])
        a = {sl: run_variant(cfg, slippage=sl, symbols=target, strategy=h["strategy"],
                             journal_kind=None, section=SEC, **pk) for sl in slips}
        b = {sl: run_fixed(cfg, kind="timing", slippage=sl, symbols=target, **pk) for sl in slips}
        noise.append({"perm": i, "s4f_candidate": assess(cfg, a[0.0], a[hi])["candidate"],
                      "timing_candidate": assess(cfg, b[0.0], b[hi])["candidate"]})
        log.info("шум: перестановка %d/%d", i + 1, n_perm)
    noise = pd.DataFrame(noise, columns=["perm", "s4f_candidate", "timing_candidate"])
    out.mkdir(parents=True, exist_ok=True)
    noise.to_csv(out / "noise.csv", index=False)
    return verdict(cfg, R, sec, noise, base)


def _year_stats(ew: pd.Series, ppy: float) -> dict:
    res = {}
    for y in sorted(set(ew.index.year)):
        x = ew[ew.index.year == y]
        eq = (1 + x).cumprod()
        res[y] = (_sharpe(x, ppy), float(eq.iloc[-1] - 1), float((eq / eq.cummax() - 1).min()))
    return res


def verdict(cfg: dict, R: dict, sec: dict, noise: pd.DataFrame, base: dict) -> Path:
    h = cfg[SEC]
    tf, ppy = h["tf"], periods_per_year(h["tf"], cfg)
    hi = max(float(s) for s in h["slippage"])
    f = lambda x, nd=2: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"
    att = {"s4f": R[("s4f", 0.0)]["ew"], "timing": R[("timing", 0.0)]["ew"]}
    l2 = level2(cfg, att)
    n_log = int((T.read_trials(cfg["paths"]["trials_log"]).kind == "attempt").sum())
    names = {"s4f": "S4 + фильтр SMA_200d", "timing": "Тайминг SMA_200d"}
    res = {}
    for k in ("s4f", "timing"):
        a = assess(cfg, R[(k, 0.0)], R[(k, hi)])
        ew = att[k]
        sr_b, sk, ku, n_obs = moments(ew.to_numpy())
        dsr = psr(sr_b, l2["sr0_bar"], n_obs=n_obs, skew=sk, kurt=ku) if n_obs > 2 else np.nan
        share = float(noise[f"{k}_candidate"].mean()) if len(noise) else np.nan
        if a["candidate"] and a["t"] >= h["t_significant"] and dsr >= h["dsr_significant"]:
            v = "Значимо"
        elif a["candidate"]:
            v = "Кандидат"
        else:
            v = "Не подтверждена"
        if a["candidate"] and share >= h["weak_share"]:
            v += " — слабо"
        res[k] = {**a, "dsr": dsr, "noise": share, "verdict": v}
    c1, c2 = res["s4f"]["candidate"], res["timing"]["candidate"]
    if c1 and c2:
        pick = "s4f" if res["s4f"]["sr"] > res["timing"]["sr"] else "timing"
    else:
        pick = "s4f" if c1 else ("timing" if c2 else None)
    L = ["# H4: вердикт (docs/preregistration_h4.md §4–§5)", "",
         f"Дата: {pd.Timestamp.now(tz='UTC'):%Y-%m-%d}, git {T.git_hash()}. Train до "
         f"{_end(cfg, SEC)}, все фолды валидации; test не открывался.", "",
         f"**Кандидат для test (§5): {names[pick] if pick else 'нет — test не открывается'}.**", "",
         "## Попытки (BTC/ETH/SOL, EW-OOS)", "",
         "| Условие | " + " | ".join(names.values()) + " | Порог |", "| --- | --- | --- | --- |",
         f"| **Вердикт** | {res['s4f']['verdict']} | {res['timing']['verdict']} | |",
         f"| EW-OOS Sharpe | {f(res['s4f']['sr'])} | {f(res['timing']['sr'])} | > 0 |",
         f"| t по Lo | {f(res['s4f']['t'])} | {f(res['timing']['t'])} | кандидат ≥ {h['t_candidate']:g}; значимо ≥ {h['t_significant']:g} |",
         f"| Sharpe при slippage {hi:g} | {f(res['s4f']['sr_slip'])} | {f(res['timing']['sr_slip'])} | > 0 |",
         f"| Торговали | {res['s4f']['n_traded']} из 3 | {res['timing']['n_traded']} из 3 | ≥ {h['min_traded']} |",
         f"| Все торговавшие > 0 | {'да' if res['s4f']['all_traded_pos'] else 'нет'} | {'да' if res['timing']['all_traded_pos'] else 'нет'} | да |",
         f"| DSR | {f(res['s4f']['dsr'])} | {f(res['timing']['dsr'])} | значимо ≥ {h['dsr_significant']:g} |",
         f"| Шум: доля перестановок, прошедших «кандидата» | {f(res['s4f']['noise'])} | {f(res['timing']['noise'])} | ≥ {h['weak_share']:g} → «слабо» |", "",
         f"DSR: N = {l2['n_attempts']} (в журнале attempt: {n_log}); N_eff = {l2['n_eff_common']} "
         f"(H1–H3 и H4 на {l2['n_common_bars']} общих 4h-барах) + 1 (H3b) = {l2['n_eff']}; "
         f"SR₀ на 4h-бар = {l2['sr0_bar']:.3e}.", ""]
    d1 = lw_sharpe_diff(att["s4f"], att["timing"], periods_per_year=ppy)
    d2 = lw_sharpe_diff(att["s4f"], R[("s4", 0.0)]["ew"], periods_per_year=ppy)
    L += ["## Не в критериях", "",
          "ΔSharpe (Ledoit–Wolf, HAC, QS):", "",
          "| Разность | ΔSR | SE | p |", "| --- | --- | --- | --- |",
          f"| S4 + фильтр − тайминг SMA | {f(d1[0])} | {f(d1[1])} | {f(d1[2], 3)} |",
          f"| S4 + фильтр − S4 без фильтра | {f(d2[0])} | {f(d2[1])} | {f(d2[2], 3)} |", "",
          "### По годам: Sharpe / доходность / MDD (EW, BTC/ETH/SOL)", ""]
    comp = [("S4 + фильтр", R[("s4f", 0.0)]["ew"]), ("Тайминг SMA", R[("timing", 0.0)]["ew"]),
            ("Buy & hold", R[("bh", 0.0)]["ew"]), ("S4 без фильтра", R[("s4", 0.0)]["ew"])]
    ys = {n: _year_stats(x, ppy) for n, x in comp}
    years = sorted({y for d in ys.values() for y in d})
    L += ["| Стратегия | " + " | ".join(str(y) for y in years) + " | Весь период Sharpe |",
          "| --- " * (len(years) + 2) + "|"]
    for n, x in comp:
        cells = [f"{f(ys[n][y][0])} / {f(100 * ys[n][y][1], 1)}% / {f(100 * ys[n][y][2], 1)}%"
                 if y in ys[n] else "—" for y in years]
        L.append(f"| {n} | " + " | ".join(cells) + f" | {f(_sharpe(x, ppy))} |")
    L += ["", "### Лонг / шорт (Sharpe EW по знаку удерживаемой позиции)", "",
          "| Стратегия | Лонг | Шорт |", "| --- | --- | --- |"]
    for k, n in (("s4f", "S4 + фильтр"), ("timing", "Тайминг SMA"), ("s4", "S4 без фильтра")):
        v = R[(k, 0.0)]
        L.append(f"| {n} | {f(_sharpe(v['ew_long'], ppy))} | {f(_sharpe(v['ew_short'], ppy))} |")
    # фильтр: доля запрета, сделки, входы посреди тренда
    L += ["", "### Фильтр: что он делает (по парам)", "",
          "| Пара | Запрет направления S4 (доля OOS-баров) | Сделок S4 → S4 + фильтр | Входы посреди тренда | OOS-Sharpe S4 + фильтр | OOS-Sharpe тайминг | OOS-Sharpe S4 | OOS-Sharpe B&H |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    pf, pt, p4, pb = (R[k]["pairs"].set_index("symbol") for k in
                      (("s4f", 0.0), ("timing", 0.0), ("s4", 0.0), ("bh", 0.0)))
    for sym in pf.index:
        df = base[sym][0]
        al = sma_filter.allowed(df, **sma_filter.config_params(tf, cfg))
        pa = R[("s4", 0.0)]["series"]["pos"][sym]
        a2 = al.loc[pa.index]
        forb = ((pa > 0) & ~a2["long"]) | ((pa < 0) & ~a2["short"])
        pfs = R[("s4f", 0.0)]["series"]["pos"][sym]
        entry = (np.sign(pfs) != 0) & (np.sign(pfs.shift(1).fillna(0)) == 0)
        mid = entry & (np.sign(pa) == np.sign(pa.shift(1).fillna(0)))
        L.append(f"| {sym} | {f(float(forb.mean()))} | {int(p4.loc[sym, 'oos_n_trades'])} → "
                 f"{int(pf.loc[sym, 'oos_n_trades'])} | {f(float(mid.sum() / entry.sum()) if entry.sum() else np.nan)} | "
                 f"{f(pf.loc[sym, 'oos_sharpe'])} | {f(pt.loc[sym, 'oos_sharpe'])} | "
                 f"{f(p4.loc[sym, 'oos_sharpe'])} | {f(pb.loc[sym, 'oos_sharpe'])} |")
    ew = R[("s4f", 0.0)]["ew"]
    qs = _quarters(ew.index)
    L += ["", "### S4 + фильтр по кварталам (EW)", "", "| Квартал | Sharpe |", "| --- | --- |"]
    L += [f"| {q} | {f(_sharpe(ew[qs == q], ppy))} |" for q in sorted(set(qs))]
    L += ["", f"Центр плато S4 + фильтр: {f(_sharpe(R[('s4f', 0.0)]['ew_plateau'], ppy))}.", "",
          "### Вторично: 10 пар H2 (не в критериях)", ""]
    sp = sec["s4f"]["pairs"]
    tr = sp[sp["traded"]]
    L += [f"S4 + фильтр: EW {f(_sharpe(sec['s4f']['ew'], ppy))}; торговали {len(tr)} из {len(sp)}, "
          f"из них > 0: {int((tr['oos_sharpe'] > 0).sum())}. Тайминг SMA: EW "
          f"{f(_sharpe(sec['timing']['ew'], ppy))}. Buy & hold: EW {f(_sharpe(sec['bh']['ew'], ppy))}."]
    path = Path(cfg["paths"]["h4_verdict"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.cv.h4")
    p.add_argument("--config", default="config.yaml")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(name)s: %(message)s")
    path = run(load_config(a.config))
    log.info("вердикт записан: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
