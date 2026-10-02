"""Вердикты этапа 5 по §7 предрегистрации и сводный отчёт.

Уровень 2 (§8): попытки — EW-OOS ряды 1h, slippage 0, по вариантам
{expanding, rolling} × {ансамбль, центр плато} (+ S6 с 2022); N_eff — число
собственных значений на neff_share следа корреляции торговавших (общие бары)
плюс один кластер на все попытки с нулевым рядом; V — дисперсия Sharpe на бар
торговавших; SR₀ = E[max]. DSR основного варианта стратегии
(expanding, ансамбль, slippage 0) = PSR(SR₀).

Вердикт по стратегии (1h, основной вариант):
- Значимо: EW-OOS Sharpe > 0, t по Lo ≥ 2, DSR ≥ 0.95;
- Кандидат: EW-OOS Sharpe > 0; знак сохраняется при slippage 0.0005;
  val-Sharpe > 0 больше чем в половине торговавших фолдов хотя бы на двух
  парах;
- Нет — иначе. Пара с < oos_min_trades сделок OOS — «недостаточно данных».
Строка шума: фактическая доля фолдов без торговли (среднее по парам) против
перестановочной; «отличим от шума» — если ниже 5-го перцентиля перестановок.

Итог — paths.stage5_verdicts (docs/stage5_verdicts.md, в git) и
paths.stage5_out/oos.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import periods_per_year
from src.stats.dsr import expected_max_sharpe, moments, psr
from src.stats.neff import corr_of_columns, effective_number, participation_ratio
from src.stats.sharpe import sharpe_se_lo
from src.stats.trials import git_hash


def _root(cfg):
    from src.cv.stage5 import out_root
    return out_root(cfg)


def _vid(tf, scheme, slip, s6=False):
    from src.cv.stage5 import variant_id
    return variant_id(tf, scheme, slip, s6)


def _ew(root: Path, vid: str, strategy: str, rule: str) -> pd.Series | None:
    f = root / vid / "ew" / f"{strategy}_{rule}.parquet"
    return pd.read_parquet(f)["net"] if f.exists() else None


def level2(cfg: dict) -> dict:
    """N_eff, V, SR₀ уровня 2 по всем попыткам H1 (1h, slippage 0)."""
    series, idle = level2_series(cfg)
    return level2_from_series(series, idle, cfg["selection"]["neff_share"])


def level2_series(cfg: dict) -> tuple[dict[str, pd.Series], list[str]]:
    """EW-OOS ряды попыток H1: (торговавшие {ключ: ряд}, ключи нулевых рядов).
    Нет результатов этапа 5 — пустые."""
    from src.cv.stage5 import STRATEGIES, rules_for
    root, tf = _root(cfg), cfg["stage5"]["tfs"][0]
    series: dict[str, pd.Series] = {}
    idle: list[str] = []
    for scheme in cfg["stage5"]["schemes"]:
        for s6 in (False, True):
            vid = _vid(tf, scheme, 0.0, s6)
            for strategy in (["s6_vwap"] if s6 else STRATEGIES):
                for rule in rules_for(strategy):
                    s = _ew(root, vid, strategy, rule)
                    if s is None:
                        continue
                    key = f"{vid}|{strategy}|{rule}"
                    if s.std() > 0:
                        series[key] = s
                    else:
                        idle.append(key)
    return series, idle


def level2_from_series(series: dict[str, pd.Series], idle: list[str], share: float) -> dict:
    """N_eff, V, SR₀ уровня 2.

    Правило (запись «Изменения» 2026-10-02 (б)): попытки с нулевым OOS-рядом
    (стратегия не торговала) не исключаются из N, а считаются одним кластером:
    N_eff = N_eff(торговавших) + 1, если такие есть. V — дисперсия Sharpe
    на бар только торговавших (у нулевого ряда Sharpe не определён).
    """
    if series:
        mat = pd.concat(series, axis=1, join="inner", sort=True)
        c = corr_of_columns(mat.to_numpy())
        sr_bar = mat.mean() / mat.std(ddof=1)
        n_eff_tr = effective_number(c, share)
        pr = participation_ratio(c)
        v = float(sr_bar.var(ddof=1)) if len(sr_bar) > 1 else 0.0
        n_obs = len(mat)
    else:
        n_eff_tr, pr, v, n_obs = 0, np.nan, 0.0, 0
    n_eff = n_eff_tr + (1 if idle else 0)
    return {"n_attempts": len(series) + len(idle), "n_traded": len(series), "n_idle": len(idle),
            "traded": list(series), "idle": list(idle), "n_eff_traded": n_eff_tr,
            "n_eff": n_eff, "pr": pr, "var_sr_bar": v,
            "sr0_bar": expected_max_sharpe(n_eff, v), "n_obs": n_obs}


def _noise(root: Path) -> pd.DataFrame | None:
    f = root / "noise" / "noise_perm.csv"
    return pd.read_csv(f) if f.exists() else None


def strategy_rows(cfg: dict, tf: str, l2: dict | None) -> pd.DataFrame:
    """Вердикты по стратегиям для ТФ (основной вариант)."""
    from src.cv.stage5 import STRATEGIES
    root = _root(cfg)
    st5 = cfg["stage5"]
    ppy = periods_per_year(tf, cfg)
    prim = _vid(tf, "expanding", 0.0)
    pairs = pd.read_csv(root / prim / "oos_pairs.csv")
    noise = _noise(root) if tf == st5["tfs"][0] else None
    rows = []
    for strategy in STRATEGIES:
        ew = _ew(root, prim, strategy, "ensemble")
        pr = pairs[(pairs.strategy == strategy) & (pairs.rule == "ensemble")]
        row = {"tf": tf, "strategy": strategy}
        if ew is None or not ew.std() > 0:
            row.update(verdict="Нет", note="не торговала")
            rows.append(row)
            continue
        sr, se = sharpe_se_lo(ew.to_numpy(), ppy)
        sr_b, sk, ku, t = moments(ew.to_numpy())
        dsr = psr(sr_b, l2["sr0_bar"], n_obs=t, skew=sk, kurt=ku) if l2 else np.nan
        signs = {}
        for slip in st5["slippage"]:
            e = _ew(root, _vid(tf, "expanding", slip), strategy, "ensemble")
            signs[slip] = float(e.mean() / e.std(ddof=1) * np.sqrt(ppy)) if e is not None and e.std() > 0 else np.nan
        plat = _ew(root, prim, strategy, "plateau")
        n_pos_pairs = int((pr["share_val_pos_traded"] > 0.5).sum())
        insufficient = pr.loc[pr["insufficient"], "symbol"].tolist()
        share_nt = float(pr["share_no_trade"].mean())
        row.update(ew_sharpe=sr, t_lo=sr / se if se else np.nan, dsr=dsr,
                   sharpe_slip=signs, plateau_sharpe=(float(plat.mean() / plat.std(ddof=1) * np.sqrt(ppy))
                                                     if plat is not None and plat.std() > 0 else np.nan),
                   pairs_majority_pos=n_pos_pairs, insufficient=insufficient,
                   share_no_trade=share_nt, oos_trades=int(pr["oos_n_trades"].sum()))
        if noise is not None:
            nz = noise[noise.strategy == strategy].groupby("perm")["share_no_trade"].mean()
            row.update(noise_mean=float(nz.mean()), noise_p5=float(nz.quantile(0.05)),
                       noise_p95=float(nz.quantile(0.95)),
                       distinct_from_noise=bool(share_nt < nz.quantile(0.05)))
        slip_hold = signs.get(0.0005, np.nan) > 0
        if len(insufficient) == len(pr):
            verdict = "недостаточно данных"
        elif sr > 0 and row["t_lo"] >= 2 and l2 and dsr >= 0.95:
            verdict = "Значимо"
        elif sr > 0 and slip_hold and n_pos_pairs >= 2:
            verdict = "Кандидат"
        else:
            verdict = "Нет"
        row["verdict"] = verdict
        rows.append(row)
    return pd.DataFrame(rows)


def _f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def verdicts(cfg: dict) -> Path:
    """Посчитать уровень 2 и вердикты, записать docs/stage5_verdicts.md и results/stage5/oos.csv."""
    root = _root(cfg)
    st5 = cfg["stage5"]
    l2 = level2(cfg)
    tabs = [strategy_rows(cfg, st5["tfs"][0], l2)]
    for tf in st5["tfs"][1:]:
        if (root / _vid(tf, "expanding", 0.0) / "oos_pairs.csv").exists():
            tabs.append(strategy_rows(cfg, tf, None))
    allv = pd.concat(tabs, ignore_index=True)
    # oos.csv: пары основного варианта + базовая линия шума
    prim = _vid(st5["tfs"][0], "expanding", 0.0)
    pairs = pd.read_csv(root / prim / "oos_pairs.csv")
    nz = _noise(root)
    if nz is not None:
        agg = nz.groupby(["strategy", "symbol"])["share_no_trade"].agg(
            noise_share_mean="mean", noise_share_p5=lambda s: s.quantile(0.05),
            noise_share_p95=lambda s: s.quantile(0.95)).reset_index()
        pairs = pairs.merge(agg, on=["strategy", "symbol"], how="left")
    pairs.to_csv(root / "oos.csv", index=False)

    L = ["# Этап 5: вердикты (docs/preregistration.md §7)", "",
         f"Дата: {pd.Timestamp.now(tz='UTC'):%Y-%m-%d}, git {git_hash()}. Train до 2023-12-31; "
         "test не открывался.", "",
         "## Уровень 2 (DSR)", "",
         f"Попыток: {l2['n_attempts']} (торговали {l2['n_traded']}, нулевой OOS-ряд {l2['n_idle']} — "
         f"один кластер); N_eff (95% следа): {l2['n_eff']}; participation ratio торговавших: "
         f"{l2['pr']:.1f}; V(SR на бар): {l2['var_sr_bar']:.3e}; SR₀ на бар: {l2['sr0_bar']:.3e}; "
         f"общих баров: {l2['n_obs']}.", ""]
    for tf, tab in allv.groupby("tf", sort=False):
        is_main = tf == st5["tfs"][0]
        L += [f"## {tf}" + ("" if is_main else " (проверка, не для выбора; DSR не считается)"), "",
              "| Стратегия | EW-OOS Sharpe | t (Lo) | DSR | Sharpe при slippage 0 / 0.05% / 0.1% | "
              "Центр плато | Пар с > 50% положит. фолдов | Без торговли (факт / шум p5–p95) | "
              "Сделок OOS | Вердикт |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in tab.iterrows():
            if "ew_sharpe" not in r or pd.isna(r.get("ew_sharpe")):
                L.append(f"| {r['strategy']} | — | — | — | — | — | — | — | — | {r['verdict']} |")
                continue
            sl = r["sharpe_slip"]
            noise = (f"{_f(r['share_no_trade'])} / {_f(r.get('noise_p5'))}–{_f(r.get('noise_p95'))}"
                     if "noise_p5" in r and not pd.isna(r.get("noise_p5")) else _f(r["share_no_trade"]))
            ins = f" (мало сделок: {', '.join(r['insufficient'])})" if r["insufficient"] else ""
            L.append(f"| {r['strategy']} | {_f(r['ew_sharpe'])} | {_f(r['t_lo'])} | {_f(r['dsr'])} | "
                     f"{' / '.join(_f(sl.get(s)) for s in st5['slippage'])} | {_f(r['plateau_sharpe'])} | "
                     f"{r['pairs_majority_pos']} | {noise} | {r['oos_trades']} | {r['verdict']}{ins} |")
        if is_main and "distinct_from_noise" in tab:
            L += ["", "Отбор на train против шума (доля фолдов без торговли ниже 5-го перцентиля "
                  "перестановок):", ""]
            for _, r in tab.iterrows():
                if "distinct_from_noise" in r and not pd.isna(r.get("distinct_from_noise")):
                    L.append(f"- {r['strategy']}: отбор на train "
                             f"{'отличим от шума' if r['distinct_from_noise'] else 'не отличим от шума'}.")
        L.append("")
    path = Path(cfg["paths"]["stage5_verdicts"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    allv.to_csv(root / "verdicts.csv", index=False)
    return path
