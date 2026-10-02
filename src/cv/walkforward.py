"""Walk-forward подхода A (docs/preregistration.md §4, §6, §8).

Для каждого фолда, стратегии и пары отдельно:
1. фильтры на train-окне: cost_ratio ≤ selection.cost_ratio_max,
   trades_per_year ≥ min_trades_per_year[tf], Sharpe определён → N_pass;
2. ранжирование по годовому train-Sharpe чистой доходности;
3. top-k = min(top_k_max, ⌈N_pass / top_k_div⌉);
4. уровень 1 DSR: N_eff — число собственных значений на neff_share следа
   корреляции train-доходностей N_pass конфигураций; V — дисперсия их
   Sharpe на бар; SR₀ = E[max]; train-DSR = PSR(SR₀) лучшей;
   торговать, только если train-DSR ≥ train_dsr_min; N_pass = 0 — нет;
5. ансамбль: среднее sign(pos) по top-k; альтернатива — центр плато
   (argmax train-Sharpe, сглаженного по соседям ±1 шаг по осям сетки среди
   прошедших фильтры); правило отказа то же.

Всё, по чему выбираем, считается только на train-окне [train_start, val_start).
Сигналы — один раз по всей истории (grid.build_grid); окна вырезаются после.
Позиции кварталов валидации сшиваются в один ряд на пару и прогоняются
движком целиком: смена ансамбля на стыке облагается комиссией, позиция на
первом баре квартала — решение с close последнего бара предыдущего.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest import engine
from src.backtest.metrics import metrics
from src.cv.grid import GridData, mask_stats, window_rows, window_stats
from src.stats.dsr import expected_max_sharpe, moments, psr
from src.stats.neff import corr_of_columns, effective_number, participation_ratio


@dataclass(frozen=True)
class Fold:
    """Один фолд: обучение [train_start, train_end), валидация [val_start, val_end)."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def _align_up(ts: pd.Timestamp, step_months: int) -> pd.Timestamp:
    """Ближайшее начало периода (месяцы 1, 1+step, …) не раньше ts, 00:00 UTC."""
    start = pd.Timestamp(year=ts.year, month=1, day=1, tz=ts.tz)
    while start < ts:
        start = start + pd.DateOffset(months=step_months)
    return start


def folds(index: pd.DatetimeIndex, *, scheme: str, step_months: int,
          min_train_months: int, train_end: pd.Timestamp,
          usable_from: pd.Timestamp | None = None,
          rolling_train_months: int | None = None,
          allow_test: bool = False) -> list[Fold]:
    """Построить фолды по календарным периодам (шаг step_months, для квартала — 3).

    Начало данных — usable_from пары (universe.csv; данные раньше — только
    прогрев индикаторов), иначе первый бар index. Первая валидация — первое
    начало периода не раньше начала данных + min_train_months.
    expanding: train = [начало данных, val_start);
    rolling: train = [max(начало данных, val_start − rolling_train_months), val_start).
    Валидация — [val_start, val_start + step_months).

    train_end — ИСКЛЮЧИТЕЛЬНАЯ граница train этапа (2024-01-01 для train до
    2023-12-31): без allow_test ни одно окно валидации её не пересекает; с
    allow_test фолды продолжаются до конца index.

    Raises:
        ValueError: неизвестная схема, нет rolling_train_months для rolling,
            данных меньше min_train_months до train_end.
    """
    if scheme not in ("expanding", "rolling"):
        raise ValueError(f"неизвестная схема: {scheme}")
    if scheme == "rolling" and not rolling_train_months:
        raise ValueError("для rolling нужен rolling_train_months")
    data_start = index[0] if usable_from is None else max(index[0], usable_from)
    limit = train_end if not allow_test else index[-1] + pd.Timedelta(microseconds=1)
    val_start = _align_up(data_start + pd.DateOffset(months=min_train_months), step_months)
    out: list[Fold] = []
    while True:
        val_end = val_start + pd.DateOffset(months=step_months)
        if val_end > limit and not (allow_test and val_start < limit):
            break
        if scheme == "expanding":
            tr_start = data_start
        else:
            tr_start = max(data_start, val_start - pd.DateOffset(months=rolling_train_months))
        out.append(Fold(tr_start, val_start, val_start, val_end))
        val_start = val_end
    if not out:
        raise ValueError("данных меньше min_train_months до train_end")
    return out


@dataclass
class FoldSelection:
    """Решение по одному фолду."""

    fold: Fold
    train_start: pd.Timestamp       # с учётом train_from варианта
    n_grid: int
    n_excl_cost: int
    n_excl_trades: int
    n_pass: int
    k: int
    topk: list[int]
    plateau: int | None
    best: int | None
    best_sharpe: float
    n_eff: int
    pr: float
    sr0_bar: float
    train_dsr: float
    trade: bool
    train_sharpe: np.ndarray = field(repr=False)   # годовой, по всем конфигурациям
    passed: np.ndarray = field(repr=False)


def _plateau(coords: np.ndarray, sharpe: np.ndarray, passed: np.ndarray) -> int | None:
    """argmax train-Sharpe, сглаженного по соседям (Чебышёв ≤ 1) среди прошедших."""
    idx = np.flatnonzero(passed)
    if len(idx) == 0:
        return None
    if coords.shape[1] == 0:
        return int(idx[np.argmax(sharpe[idx])])
    best, best_val = None, -np.inf
    for j in idx:
        nb = idx[np.abs(coords[idx] - coords[j]).max(axis=1) <= 1]
        v = float(np.mean(sharpe[nb]))
        if v > best_val:
            best, best_val = int(j), v
    return best


def select_rows(gd: GridData, rows: np.ndarray, cfg: dict, *,
                periods_per_year: float) -> dict:
    """Отбор §4 на произвольной bool-маске строк train (для фолдов и CPCV).

    Returns:
        словарь полей FoldSelection без fold/train_start.
    """
    sel = cfg["selection"]
    st = mask_stats(gd, rows, periods_per_year=periods_per_year)
    sh = st["sharpe"].to_numpy()
    cr = st["cost_ratio"].to_numpy()
    tpy = st["trades_per_year"].to_numpy()
    ok_cost = ~(cr > sel["cost_ratio_max"])                 # NaN (нет показателя) — проходит
    ok_trades = tpy >= sel["min_trades_per_year"][gd.tf]
    passed = ok_cost & ok_trades & ~np.isnan(sh)
    n_pass = int(passed.sum())
    idx = np.flatnonzero(passed)
    k = min(int(sel["top_k_max"]), math.ceil(n_pass / sel["top_k_div"])) if n_pass else 0
    order = idx[np.argsort(-sh[idx], kind="stable")]
    topk = [int(i) for i in order[:k]]
    best = topk[0] if topk else None
    n_eff, pr, sr0, tdsr = 0, np.nan, np.nan, np.nan
    if n_pass:
        r = np.flatnonzero(rows)
        x = gd.net[r][:, idx]
        if n_pass > 1:
            c = corr_of_columns(x)
            n_eff = effective_number(c, sel["neff_share"])
            pr = participation_ratio(c)
            v = float(np.var(st["sr_bar"].to_numpy()[idx], ddof=1))
        else:
            n_eff, pr, v = 1, 1.0, 0.0
        sr0 = expected_max_sharpe(n_eff, v)
        sr_b, sk, ku, t = moments(gd.net[r][:, best])
        tdsr = psr(sr_b, sr0, n_obs=t, skew=sk, kurt=ku)
    trade = bool(n_pass > 0 and not np.isnan(tdsr) and tdsr >= sel["train_dsr_min"])
    return dict(n_grid=len(sh), n_excl_cost=int((~ok_cost).sum()),
                n_excl_trades=int((ok_cost & ~ok_trades).sum()), n_pass=n_pass, k=k,
                topk=topk, plateau=_plateau(gd.coords, sh, passed), best=best,
                best_sharpe=float(sh[best]) if best is not None else np.nan,
                n_eff=n_eff, pr=pr, sr0_bar=sr0, train_dsr=tdsr, trade=trade,
                train_sharpe=sh, passed=passed)


def select_fold(gd: GridData, fold: Fold, cfg: dict, *, periods_per_year: float,
                train_from: pd.Timestamp | None = None) -> FoldSelection:
    """Отбор конфигураций на train-окне фолда (только данные до val_start)."""
    t0 = fold.train_start if train_from is None else max(fold.train_start, train_from)
    rows = np.zeros(len(gd.index), dtype=bool)
    rows[window_rows(gd.index, t0, fold.train_end)] = True
    return FoldSelection(fold=fold, train_start=t0,
                         **select_rows(gd, rows, cfg, periods_per_year=periods_per_year))


@dataclass
class WFResult:
    """Итог walk-forward одной стратегии на одной паре."""

    selections: list[FoldSelection]
    bt_ensemble: pd.DataFrame       # движок по сшитому ряду (вся история)
    bt_plateau: pd.DataFrame
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    folds_table: pd.DataFrame       # по фолду: отбор и метрики валидации
    scatter: pd.DataFrame           # train- и val-Sharpe всех конфигураций по фолдам


def walk_forward(gd: GridData, df: pd.DataFrame, funding: pd.Series | None,
                 fold_list: list[Fold], cfg: dict, *, slippage: float,
                 periods_per_year: float,
                 train_from: pd.Timestamp | None = None,
                 tradable: dict | None = None) -> WFResult:
    """Отбор по фолдам, сшивка позиций валидации, метрики по кварталам и итогу.

    tradable — необязательная карта {val_start: bool} (фильтр торгуемости H2):
    False — позиция 0 на весь квартал независимо от отбора; в таблице фолдов
    колонка liquidity_ok.
    """
    from src.data.universe import usable_from

    n = len(gd.index)
    pe = np.zeros(n)
    pp = np.zeros(n)
    sels, scat = [], []
    for f in fold_list:
        s = select_fold(gd, f, cfg, periods_per_year=periods_per_year, train_from=train_from)
        sels.append(s)
        rows = window_rows(gd.index, f.val_start, f.val_end)
        liq_ok = True if tradable is None else bool(tradable.get(f.val_start, False))
        if s.trade and liq_ok:
            pe[rows] = np.sign(gd.pos[rows][:, s.topk]).mean(axis=1)
            pp[rows] = np.sign(gd.pos[rows][:, s.plateau])
        vs = window_stats(gd, f.val_start, f.val_end, periods_per_year=periods_per_year)
        scat.append(pd.DataFrame({"fold": str(f.val_start.date()), "config": np.arange(len(vs)),
                                  "train_sharpe": s.train_sharpe, "val_sharpe": vs["sharpe"],
                                  "passed": s.passed}))
    c = cfg["costs"]
    kw = dict(fee_per_side=float(c["fee_per_side"]), slippage_per_side=slippage,
              funding=funding if c.get("include_funding", True) else None, tf=gd.tf,
              active_from=usable_from(cfg, gd.symbol))
    bte = engine.run(df, pd.Series(pe, index=df.index), **kw)
    btp = engine.run(df, pd.Series(pp, index=df.index), **kw)
    rows_out = []
    for s in sels:
        f = s.fold
        mask = (df.index >= f.val_start) & (df.index < f.val_end)
        me = metrics(bte, periods_per_year=periods_per_year, mask=mask)
        mp = metrics(btp, periods_per_year=periods_per_year, mask=mask)
        rows_out.append({
            "symbol": gd.symbol, "strategy": gd.strategy, "tf": gd.tf,
            "val_start": f.val_start, "val_end": f.val_end, "train_start": s.train_start,
            "n_grid": s.n_grid, "n_excl_cost": s.n_excl_cost, "n_excl_trades": s.n_excl_trades,
            "n_pass": s.n_pass, "k": s.k, "n_eff": s.n_eff, "pr": s.pr,
            "best_train_sharpe": s.best_sharpe, "sr0_bar": s.sr0_bar, "train_dsr": s.train_dsr,
            "trade": s.trade,
            "liquidity_ok": True if tradable is None else bool(tradable.get(f.val_start, False)),
            "topk_params": [gd.grid_params[i] for i in s.topk],
            "plateau_params": gd.grid_params[s.plateau] if s.plateau is not None else None,
            **{f"ens_{k}": v for k, v in me.items()},
            **{f"plt_{k}": v for k, v in mp.items()},
        })
    return WFResult(sels, bte, btp, fold_list[0].val_start, fold_list[-1].val_end,
                    pd.DataFrame(rows_out),
                    pd.concat(scat, ignore_index=True).assign(symbol=gd.symbol,
                                                              strategy=gd.strategy, tf=gd.tf))
