"""CSCV/PBO (Bailey et al., 2017) и CPCV (López de Prado, AFML гл. 12). Этап 6.

Диагностика: без вердиктов, без test, в N для DSR не входит.

Данные — чистые доходности конфигураций сетки на train (кэш этапа 5:
комиссия + funding, slippage 0). Позиции непрерывны, сделки длинные, поэтому
между IS и OOS — embargo: из IS (train) удаляются бары в пределах h от стыка
с OOS (test) группой; h = максимальная по сетке средняя длительность сделки
на train в барах (ограничена половиной группы).

CSCV: S последовательных групп, все C(S, S/2) разбиений IS/OOS. Для каждого:
n* = argmax IS-Sharpe; ω = ранг n* по OOS-Sharpe / (N + 1); λ = ln(ω/(1−ω));
PBO = доля λ ≤ 0. Рядом: P(OOS-Sharpe(n*) < 0), медиана OOS-Sharpe(n*),
наклон регрессии OOS-Sharpe(n*) на IS-Sharpe(n*). Статистики разбиения
собираются из сумм по ядру группы и краевым полосам h (без прохода по барам).

CPCV: N групп, k тестовых → C(N, k) разбиений, φ = k·C(N, k)/N путей; каждая
группа в каждом пути тестируется ровно один раз. На каждом разбиении отбор
на train-группах (с embargo): (i) процедура этапа 5 целиком (фильтры, top-k,
отказ по train-DSR); (ii) лучшая по IS-Sharpe среди прошедших фильтры, без
правила отказа. Пути склеиваются из тестовых кусков и прогоняются движком
целиком (смена выбора на стыке групп облагается комиссией).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from src.backtest import engine
from src.cv.grid import GridData


# ── группы и embargo ─────────────────────────────────────────────────────

def group_bounds(n_rows: int, n_groups: int) -> list[tuple[int, int]]:
    """Границы [a, b) последовательных почти равных групп."""
    edges = np.linspace(0, n_rows, n_groups + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def embargo_bars(pos: np.ndarray) -> int:
    """Максимальная по конфигурациям средняя длительность сделки (бары удержания / входы)."""
    s = np.sign(pos)
    prev = np.vstack([np.zeros((1, s.shape[1])), s[:-1]])
    entries = ((s != 0) & (s != prev)).sum(axis=0)
    held = (s != 0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dur = np.where(entries > 0, held / entries, 0.0)
    return int(np.ceil(dur.max())) if len(dur) else 0


def _block_sums(x: np.ndarray, bounds, h: int):
    """Суммы (n, Σx, Σx²) по ядру, левой и правой полосам h каждой группы."""
    k = x.shape[1]
    out = {name: np.zeros((len(bounds), 3, k)) for name in ("core", "left", "right")}
    for g, (a, b) in enumerate(bounds):
        hh = min(h, (b - a) // 2)
        parts = {"left": x[a:a + hh], "core": x[a + hh:b - hh], "right": x[b - hh:b]}
        for name, seg in parts.items():
            out[name][g, 0] = len(seg)
            out[name][g, 1] = seg.sum(axis=0)
            out[name][g, 2] = (seg ** 2).sum(axis=0)
    return out


def _sharpe_from_sums(s: np.ndarray) -> np.ndarray:
    """SR на бар из (n, Σx, Σx²) — по последней оси конфигураций; s: (..., 3, k)."""
    n, s1, s2 = s[..., 0, :], s[..., 1, :], s[..., 2, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = s1 / n
        var = (s2 - n * mu ** 2) / (n - 1)
        return np.where(var > 0, mu / np.sqrt(var), np.nan)


def _is_masks(in_is: np.ndarray):
    """По матрице принадлежности IS (combos × S): включать ли левую/правую полосу группы."""
    s = in_is.shape[1]
    left_ok = in_is.copy()
    right_ok = in_is.copy()
    left_ok[:, 1:] &= in_is[:, :-1]          # левый сосед тоже IS (или группа первая)
    right_ok[:, :-1] &= in_is[:, 1:]         # правый сосед тоже IS (или группа последняя)
    return left_ok, right_ok


# ── CSCV / PBO ───────────────────────────────────────────────────────────

def cscv_combos(n_groups: int) -> np.ndarray:
    """Все разбиения на IS (половина групп): bool-матрица combos × S."""
    if n_groups % 2:
        raise ValueError("S должно быть чётным")
    combos = list(itertools.combinations(range(n_groups), n_groups // 2))
    m = np.zeros((len(combos), n_groups), dtype=bool)
    for i, c in enumerate(combos):
        m[i, list(c)] = True
    return m


def cscv(net: np.ndarray, *, n_groups: int, h: int, periods_per_year: float) -> dict:
    """PBO и сопутствующие величины по матрице доходностей train (T × C)."""
    t, k = net.shape
    bounds = group_bounds(t, n_groups)
    blk = _block_sums(net, bounds, h)
    in_is = cscv_combos(n_groups)
    left_ok, right_ok = _is_masks(in_is)
    full = blk["core"] + blk["left"] + blk["right"]                      # (S, 3, k)
    # IS: ядро IS-групп + полосы, если сосед со стороны полосы тоже IS
    is_s = (np.einsum("cg,gjk->cjk", in_is.astype(float), blk["core"])
            + np.einsum("cg,gjk->cjk", left_ok.astype(float), blk["left"])
            + np.einsum("cg,gjk->cjk", right_ok.astype(float), blk["right"]))
    oos_s = np.einsum("cg,gjk->cjk", (~in_is).astype(float), full)
    is_sr = _sharpe_from_sums(is_s)                                      # combos × k
    oos_sr = _sharpe_from_sums(oos_s)
    best = np.nanargmax(np.where(np.isnan(is_sr), -np.inf, is_sr), axis=1)
    rows = np.arange(len(best))
    # ранг n* по OOS (1 = худший), ω = ранг / (N + 1)
    oos_filled = np.where(np.isnan(oos_sr), -np.inf, oos_sr)
    rank = (oos_filled < oos_filled[rows, best][:, None]).sum(axis=1) + 1
    omega = rank / (k + 1)
    lam = np.log(omega / (1 - omega))
    sr_is_b, sr_oos_b = is_sr[rows, best], oos_sr[rows, best]
    ok = ~np.isnan(sr_is_b) & ~np.isnan(sr_oos_b)
    slope = (np.polyfit(sr_is_b[ok], sr_oos_b[ok], 1)[0] if ok.sum() > 2 and k > 1 else np.nan)
    is_bars = (in_is * np.array([b - a for a, b in bounds])).sum(axis=1)
    kept = is_s[:, 0, 0]
    ann = np.sqrt(periods_per_year)
    return {"pbo": float((lam <= 0).mean()), "lambdas": lam,
            "p_oos_loss": float((sr_oos_b < 0).mean()),
            "median_oos_sharpe": float(np.nanmedian(sr_oos_b) * ann),
            "median_is_sharpe": float(np.nanmedian(sr_is_b) * ann),
            "degradation_slope": float(slope),
            "embargo_share": float(np.mean(1 - kept / is_bars)),
            "n_configs": k, "n_combos": len(best), "h": h, "n_groups": n_groups,
            "best_counts": np.bincount(best, minlength=k)}


# ── CPCV ─────────────────────────────────────────────────────────────────

def cpcv_splits(n_groups: int, k_test: int) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Разбиения (тестовые группы) и назначение путей: paths[p, g] = индекс разбиения,
    в котором группа g тестируется в пути p."""
    splits = list(itertools.combinations(range(n_groups), k_test))
    n_paths = k_test * len(splits) // n_groups
    paths = np.full((n_paths, n_groups), -1, dtype=int)
    seen = np.zeros(n_groups, dtype=int)
    for i, sp in enumerate(splits):
        for g in sp:
            paths[seen[g], g] = i
            seen[g] += 1
    return splits, paths


def cpcv(gd: GridData, df: pd.DataFrame, funding: pd.Series | None, cfg: dict, *,
         start: pd.Timestamp, end: pd.Timestamp, n_groups: int, k_test: int, h: int,
         periods_per_year: float) -> pd.DataFrame:
    """OOS-Sharpe путей CPCV для правил (i) «процедура этапа 5» и (ii) «лучшая IS».

    Returns:
        строки: rule, path, sharpe, n_obs, share_traded_splits; плюс атрибут
        attrs["path_net"] = {(rule, path): pd.Series} для агрегатов.
    """
    from src.cv.walkforward import select_rows
    from src.data.universe import usable_from

    a0 = int(gd.index.searchsorted(start))
    b0 = int(gd.index.searchsorted(end))
    bounds = [(a0 + a, a0 + b) for a, b in group_bounds(b0 - a0, n_groups)]
    splits, paths = cpcv_splits(n_groups, k_test)
    n = len(gd.index)
    test_pos = {"procedure": [], "best_is": []}
    traded = []
    for sp in splits:
        train = np.zeros(n, dtype=bool)
        for g, (a, b) in enumerate(bounds):
            if g in sp:
                continue
            hh = min(h, (b - a) // 2)
            lo = a + hh if (g - 1) in sp else a
            hi = b - hh if (g + 1) in sp else b
            train[lo:hi] = True
        s = select_rows(gd, train, cfg, periods_per_year=periods_per_year)
        traded.append(s["trade"])
        pe = np.zeros(n)
        pb = np.zeros(n)
        for g in sp:
            a, b = bounds[g]
            if s["trade"]:
                pe[a:b] = np.sign(gd.pos[a:b][:, s["topk"]]).mean(axis=1)
            if s["best"] is not None:
                pb[a:b] = np.sign(gd.pos[a:b, s["best"]])
        test_pos["procedure"].append(pe)
        test_pos["best_is"].append(pb)
    c = cfg["costs"]
    kw = dict(fee_per_side=float(c["fee_per_side"]), slippage_per_side=0.0,
              funding=funding if c.get("include_funding", True) else None, tf=gd.tf,
              active_from=usable_from(cfg, gd.symbol))
    rows, nets = [], {}
    for rule, plist in test_pos.items():
        for p in range(paths.shape[0]):
            pos = np.zeros(n)
            for g, (a, b) in enumerate(bounds):
                pos[a:b] = plist[paths[p, g]][a:b]
            bt = engine.run(df, pd.Series(pos, index=df.index), **kw)
            x = bt["net"].iloc[a0:b0]
            sd = x.std(ddof=1)
            nets[(rule, p)] = x
            rows.append({"rule": rule, "path": p, "n_obs": len(x),
                         "sharpe": float(x.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else np.nan,
                         "share_traded_splits": float(np.mean(traded))})
    out = pd.DataFrame(rows)
    out.attrs["path_net"] = nets
    return out
