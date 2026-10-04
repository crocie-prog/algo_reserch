"""Прогон гипотезы H3b по предрегистрации (docs/preregistration_h3b.md).

S4-4h с протоколом H3 без изменений (сетка, фильтры, ансамбль top-k, отказ
при train-DSR < 0.5, фильтр торгуемости), expanding-train до начала квартала,
валидация — только кварталы 2024 г. Целевая вселенная — universe.symbols
(BTC/ETH/SOL); вторичная — h2.symbols (diagnostic).

Вердикт (§4): EW-OOS 2024 по целевым парам > 0, знак при slippage 0.0005,
торговали ≥ min_traded, все торговавшие > 0. t и DSR выводятся, в вердикт не
входят; DSR: N сквозное, H3b — отдельный кластер (N_eff(H1–H3) + 1).
Шум: доля перестановок суток, прошедших критерий; ≥ weak_share →
«слабое подтверждение».

    python -m src.cv.h3b
В stdout — только ход прогона, без метрик.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.metrics import metrics
from src.config import load_config, periods_per_year
from src.cv import report
from src.cv.grid import build_grid
from src.cv.h2 import _sharpe, _tradable_map
from src.cv.h3 import to_tf
from src.cv.walkforward import folds, walk_forward
from src.data.load import load, load_funding
from src.data.universe import usable_from
from src.stats import trials as T
from src.stats.dsr import expected_max_sharpe, moments, psr
from src.stats.sharpe import sharpe_se_lo

log = logging.getLogger("h3b")


def _end(cfg: dict, section: str = "h3b") -> str:
    t = pd.Timestamp(cfg[section]["train_end_exclusive"]) - pd.Timedelta(days=1)
    return t.date().isoformat()


def _load_pair(cfg: dict, symbol: str, tf: str, section: str = "h3b"):
    e = _end(cfg, section)
    return load(symbol, tf, end=e, cfg=cfg), load_funding(symbol, end=e, cfg=cfg)


def fold_list(cfg: dict, section: str, symbol: str, index: pd.DatetimeIndex) -> list:
    """Фолды гипотезы: expanding до train_end_exclusive; при val_from — только
    кварталы валидации не раньше val_from."""
    h, wfc = cfg[section], cfg["walk_forward"]
    fl = folds(index, scheme=h["scheme"], step_months=wfc["step_months"],
               min_train_months=wfc["min_train_months"],
               train_end=pd.Timestamp(h["train_end_exclusive"], tz="UTC"),
               usable_from=usable_from(cfg, symbol),
               rolling_train_months=wfc["rolling_train_months"])
    if h.get("val_from"):
        fl = [f for f in fl if f.val_start >= pd.Timestamp(h["val_from"], tz="UTC")]
    return fl


def _quarters(index: pd.DatetimeIndex) -> np.ndarray:
    """Метки кварталов UTC вида 2024Q1."""
    return np.asarray(index.tz_convert(None).to_period("Q").astype(str))


def _ew(series: dict[str, pd.Series]) -> pd.Series:
    if not series:
        return pd.Series(dtype=float)
    return pd.concat(series, axis=1, sort=True).mean(axis=1, skipna=True).dropna()


def run_variant(cfg: dict, *, slippage: float, symbols: list[str], tradable_fn=None,
                data_fn=None, journal_kind: str | None = "train", section: str = "h3b",
                strategy: str | None = None) -> dict:
    """Один прогон walk-forward с отбором: OOS-ряды пар (net, лонг, шорт), таблицы.
    section — раздел config гипотезы (h3b, h4); strategy — вместо h[strategy]."""
    h = cfg[section]
    tf, ppy = h["tf"], periods_per_year(h["tf"], cfg)
    strategy = strategy or h["strategy"]
    vid = f"{section}_{strategy}_{tf}_{h['scheme']}_slip{slippage:g}"
    tradable_fn = tradable_fn or _tradable_map
    out = {"net": {}, "long": {}, "short": {}, "plateau": {}, "held": {}, "pos": {}}
    pairs, ftabs = [], []
    for sym in symbols:
        df, fund = data_fn(sym) if data_fn else _load_pair(cfg, sym, tf, section)
        fl = fold_list(cfg, section, sym, df.index)
        gd = build_grid(df, fund, strategy=strategy, tf=tf, symbol=sym, cfg=cfg,
                        slippage=slippage)
        wf = walk_forward(gd, df, fund, fl, cfg, slippage=slippage, periods_per_year=ppy,
                          tradable=tradable_fn(cfg, sym, fl))
        bt = wf.bt_ensemble
        mask = (bt.index >= wf.oos_start) & (bt.index < wf.oos_end)
        w = bt.loc[mask]
        net = w["net"].rename(sym)
        out["net"][sym] = net
        out["long"][sym] = net.where(w["held"] > 0, 0.0)
        out["short"][sym] = net.where(w["held"] < 0, 0.0)
        out["held"][sym] = w["held"]
        out["pos"][sym] = w["pos"]
        btp = wf.bt_plateau
        out["plateau"][sym] = btp.loc[mask, "net"].rename(sym)
        m = metrics(bt, periods_per_year=ppy, mask=mask)
        ft = wf.folds_table
        row = {"symbol": sym, "n_folds": len(ft), "oos_sharpe": m["sharpe"],
               "oos_n_trades": m["n_trades"], "traded": bool(m["n_trades"] > 0),
               "share_long": float((w["held"] > 0).mean()),
               "share_short": float((w["held"] < 0).mean()),
               "sharpe_long": _sharpe(out["long"][sym], ppy),
               "sharpe_short": _sharpe(out["short"][sym], ppy),
               "sum_long": float(out["long"][sym].sum()),
               "sum_short": float(out["short"][sym].sum()),
               "share_refused_dsr": float((ft["liquidity_ok"] & ~ft["trade"]).mean()),
               "share_blocked_liq": float((~ft["liquidity_ok"]).mean()),
               "plt_oos_sharpe": _sharpe(out["plateau"][sym], ppy)}
        qs = _quarters(net.index)
        for q in sorted(set(qs)):
            row[f"sharpe_{q}"] = _sharpe(net[qs == q], ppy)
        pairs.append(row)
        topk = {s.fold.val_start: json.dumps([gd.grid_params[j] for j in s.topk])
                for s in wf.selections}
        ftabs.append(ft.assign(variant=vid, topk=ft["val_start"].map(topk)))
        if journal_kind:
            T.log_rows(cfg["paths"]["trials_log"], T.train_rows(wf, gd, vid, kind=journal_kind))
        log.info("%s slip=%g: готово", sym, slippage)
    return {"vid": vid, "series": out, "ew": _ew(out["net"]), "ew_long": _ew(out["long"]),
            "ew_short": _ew(out["short"]), "ew_plateau": _ew(out["plateau"]),
            "pairs": pd.DataFrame(pairs), "folds": pd.concat(ftabs, ignore_index=True)}


def criterion(ew0: pd.Series, ew1: pd.Series, pairs: pd.DataFrame, *, ppy: float,
              min_traded: int) -> tuple[bool, dict]:
    """§4: (подтверждена?, компоненты)."""
    sr0, sr1 = _sharpe(ew0, ppy), _sharpe(ew1, ppy)
    tr = pairs[pairs["traded"]]
    c = {"sr": sr0, "sr_slip": sr1, "n_traded": int(len(tr)),
         "all_traded_pos": bool(len(tr) > 0 and (tr["oos_sharpe"] > 0).all())}
    ok = bool(sr0 > 0 and sr1 > 0 and c["n_traded"] >= min_traded and c["all_traded_pos"])
    return ok, c


def level2(cfg: dict, ew: pd.Series) -> dict:
    """Уровень 2 на 4h: H1–H3 по общим барам + H3b отдельным кластером."""
    tf = cfg["h3b"]["tf"]
    h1, idle = report.level2_series(cfg)
    series = {k: to_tf(v, tf) for k, v in h1.items()}
    idle = list(idle)
    for f in sorted(Path(cfg["paths"]["h2_out"]).glob("h2_*_slip0/ew.parquet")):
        x = pd.read_parquet(f)["net"]
        (series.__setitem__(f"h2|{f.parent.name}", to_tf(x, tf)) if x.std() > 0
         else idle.append(f"h2|{f.parent.name}"))
    for f in sorted(Path(cfg["paths"]["h3_out"]).glob("h3_*_slip0/*/ew_ensemble.parquet")):
        x = pd.read_parquet(f)["net"]
        key = f"h3|{f.parent.name}"
        series[key] = x if len(x) > 1 and x.std() > 0 else None
        if series[key] is None:
            series.pop(key)
            idle.append(key)
    base = report.level2_from_series(series, idle, cfg["selection"]["neff_share"])
    sr_bar = []
    if series:
        mat = pd.concat(series, axis=1, join="inner", sort=True)
        sr_bar = list((mat.mean() / mat.std(ddof=1)).to_numpy())
    own = float(ew.mean() / ew.std(ddof=1)) if len(ew) > 1 and ew.std() > 0 else None
    if own is not None:
        sr_bar.append(own)
    v = float(np.var(sr_bar, ddof=1)) if len(sr_bar) > 1 else 0.0
    n_eff = base["n_eff"] + 1
    return {"n_attempts": base["n_attempts"] + 1, "n_eff_prev": base["n_eff"], "n_eff": n_eff,
            "var_sr_bar": v, "sr0_bar": expected_max_sharpe(n_eff, v)}


def run(cfg: dict, *, n_perm: int | None = None, tradable_fn=None, data_fn=None,
        seed: int = 20261005) -> Path:
    """Весь прогон H3b пакетом."""
    from src.cv.permute import permute_days

    h = cfg["h3b"]
    tf, ppy = h["tf"], periods_per_year(h["tf"], cfg)
    target, second = list(cfg["universe"]["symbols"]), list(cfg["h2"]["symbols"])
    out = Path(cfg["paths"]["h3b_out"])
    res = {}
    for slip in h["slippage"]:
        slip = float(slip)
        v = run_variant(cfg, slippage=slip, symbols=target, tradable_fn=tradable_fn,
                        data_fn=data_fn, journal_kind="train" if slip == 0 else "sensitivity")
        res[slip] = v
    sec = run_variant(cfg, slippage=0.0, symbols=second, tradable_fn=tradable_fn,
                      data_fn=data_fn, journal_kind="diagnostic")
    for name, v in [*((f"target_slip{s:g}", r) for s, r in res.items()), ("secondary_slip0", sec)]:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        v["pairs"].to_csv(d / "pairs.csv", index=False)
        v["folds"].to_csv(d / "folds.csv", index=False)
        v["ew"].to_frame("net").to_parquet(d / "ew.parquet")
    v0 = res[0.0]
    rows = [
        {"kind": "attempt", "variant": v0["vid"], "symbol": f"EW{len(target)}",
         "params": {"rule": "ensemble"}, "ew": v0["ew"], "note": "target_slip0/ew.parquet"},
        {"kind": "sensitivity", "variant": res[max(res)]["vid"], "symbol": f"EW{len(target)}",
         "params": {"rule": "ensemble"}, "ew": res[max(res)]["ew"],
         "note": f"target_slip{max(res):g}/ew.parquet"},
        {"kind": "diagnostic", "variant": v0["vid"], "symbol": f"EW{len(target)}",
         "params": {"rule": "plateau"}, "ew": v0["ew_plateau"], "note": ""},
        {"kind": "diagnostic", "variant": sec["vid"], "symbol": f"EW{len(second)}_secondary",
         "params": {"rule": "ensemble"}, "ew": sec["ew"], "note": "secondary_slip0/ew.parquet"},
    ]
    T.log_rows(cfg["paths"]["trials_log"], pd.DataFrame([
        {"kind": r["kind"], "variant": r["variant"], "strategy": h["strategy"], "tf": tf,
         "symbol": r["symbol"], "fold": "", "params": r["params"],
         "sharpe": _sharpe(r["ew"], ppy), "n_obs": len(r["ew"]), "note": r["note"]} for r in rows]))
    log.info("основные варианты записаны")

    n_perm = int(n_perm if n_perm is not None else h["permutations"])
    rng = np.random.default_rng(seed)
    base = {s: (data_fn(s) if data_fn else _load_pair(cfg, s, tf)) for s in target}
    noise = []
    for i in range(n_perm):
        perm = {s: (permute_days(df, rng), f) for s, (df, f) in base.items()}
        r = {float(sl): run_variant(cfg, slippage=float(sl), symbols=target,
                                    tradable_fn=tradable_fn, data_fn=lambda s: perm[s],
                                    journal_kind=None) for sl in h["slippage"]}
        ok, c = criterion(r[0.0]["ew"], r[max(r)]["ew"], r[0.0]["pairs"], ppy=ppy,
                          min_traded=int(h["min_traded"]))
        noise.append({"perm": i, "passed": ok, **c})
        log.info("шум: перестановка %d/%d", i + 1, n_perm)
    noise = pd.DataFrame(noise, columns=["perm", "passed", "sr", "sr_slip", "n_traded",
                                         "all_traded_pos"])
    out.mkdir(parents=True, exist_ok=True)
    noise.to_csv(out / "noise.csv", index=False)
    return verdict(cfg, res, sec, noise)


def verdict(cfg: dict, res: dict, sec: dict, noise: pd.DataFrame) -> Path:
    h = cfg["h3b"]
    ppy = periods_per_year(h["tf"], cfg)
    v0, v1 = res[0.0], res[max(res)]
    ew = v0["ew"]
    ok, c = criterion(ew, v1["ew"], v0["pairs"], ppy=ppy, min_traded=int(h["min_traded"]))
    share = float(noise["passed"].mean()) if len(noise) else np.nan
    if ok:
        v = "Подтверждена" + (" — слабое подтверждение" if share >= h["weak_share"] else "")
    else:
        v = "Не подтверждена"
    sr, se = sharpe_se_lo(ew.to_numpy(), ppy)
    t = sr / se if se and se > 0 else np.nan
    l2 = level2(cfg, ew)
    sr_b, sk, ku, n_obs = moments(ew.to_numpy())
    dsr = psr(sr_b, l2["sr0_bar"], n_obs=n_obs, skew=sk, kurt=ku) if n_obs > 2 else np.nan
    n_log = int((T.read_trials(cfg["paths"]["trials_log"]).kind == "attempt").sum())
    f = lambda x, nd=2: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"
    pairs = v0["pairs"]
    L = ["# H3b: вердикт (docs/preregistration_h3b.md §4)", "",
         f"Дата: {pd.Timestamp.now(tz='UTC'):%Y-%m-%d}, git {T.git_hash()}. Train до "
         f"{_end(cfg)}, валидация — кварталы с {h['val_from']}; test не открывался.", "",
         f"**Вердикт: {v}.**", "",
         "## Критерий (BTC/ETH/SOL, EW-OOS 2024)", "",
         "| Условие | Значение | Порог |", "| --- | --- | --- |",
         f"| EW-OOS Sharpe | {f(c['sr'])} | > 0 |",
         f"| EW-OOS Sharpe при slippage {max(res):g} | {f(c['sr_slip'])} | > 0 |",
         f"| Торговали | {c['n_traded']} из {len(pairs)} | ≥ {h['min_traded']} |",
         f"| Все торговавшие > 0 | {'да' if c['all_traded_pos'] else 'нет'} | да |", "",
         f"Шум ({len(noise)} перестановок суток): прошли критерий {f(share)} "
         f"(порог «слабого подтверждения» {h['weak_share']:g}).", "",
         "## Не в вердикте", "",
         f"t по Lo: {f(t)}. DSR: {f(dsr)} (N = {l2['n_attempts']}, в журнале attempt: {n_log}; "
         f"N_eff = {l2['n_eff_prev']} (H1–H3) + 1 = {l2['n_eff']}; SR₀ на 4h-бар = {l2['sr0_bar']:.3e}).", "",
         "| Срез | EW-OOS Sharpe |", "| --- | --- |"]
    qs = _quarters(ew.index)
    for q in sorted(set(qs)):
        L.append(f"| {q} | {f(_sharpe(ew[qs == q], ppy))} |")
    L += [f"| Лонг-сторона | {f(_sharpe(v0['ew_long'], ppy))} (сумма {f(float(v0['ew_long'].sum()), 4)}) |",
          f"| Шорт-сторона | {f(_sharpe(v0['ew_short'], ppy))} (сумма {f(float(v0['ew_short'].sum()), 4)}) |",
          f"| Центр плато | {f(_sharpe(v0['ew_plateau'], ppy))} |", "",
          "Лонг/шорт — по знаку удерживаемой позиции; издержки — на бар списания.", "",
          "### По парам (целевые)", ""]
    qcols = sorted(c_ for c_ in pairs.columns if c_.startswith("sharpe_20"))
    L += ["| Пара | OOS-Sharpe | Сделок | Лонг: доля / Sharpe | Шорт: доля / Sharpe | Отказ DSR | Ликвидность | "
          + " | ".join(q.replace("sharpe_", "") for q in qcols) + " |",
          "| --- " * (7 + len(qcols)) + "|"]
    for p in pairs.itertuples():
        d = p._asdict()
        L.append(f"| {p.symbol} | {f(p.oos_sharpe)} | {int(p.oos_n_trades)} | "
                 f"{f(p.share_long)} / {f(p.sharpe_long)} | {f(p.share_short)} / {f(p.sharpe_short)} | "
                 f"{f(p.share_refused_dsr)} | {f(p.share_blocked_liq)} | "
                 + " | ".join(f(d.get(q)) for q in qcols) + " |")
    sp = sec["pairs"]
    tr = sp[sp["traded"]]
    L += ["", "### Вторично: 10 пар H2 (не в вердикте)", "",
          f"EW-OOS 2024 Sharpe {f(_sharpe(sec['ew'], ppy))}; лонг {f(_sharpe(sec['ew_long'], ppy))}, "
          f"шорт {f(_sharpe(sec['ew_short'], ppy))}. Торговали {len(tr)} из {len(sp)}; "
          f"из торговавших с OOS-Sharpe > 0: {int((tr['oos_sharpe'] > 0).sum())}.", "",
          "| Пара | OOS-Sharpe | Сделок | Отказ DSR | Ликвидность |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {p.symbol} | {f(p.oos_sharpe)} | {int(p.oos_n_trades)} | {f(p.share_refused_dsr)} | "
          f"{f(p.share_blocked_liq)} |" for p in sp.itertuples()]
    path = Path(cfg["paths"]["h3b_verdict"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.cv.h3b")
    p.add_argument("--config", default="config.yaml")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(name)s: %(message)s")
    path = run(load_config(a.config))
    log.info("вердикт записан: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
