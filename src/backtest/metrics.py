"""Метрики стратегии (как табл. 5 статьи), rf = 0.

Аннуализация — по числу баров в году для ТФ (365 дней, config.periods_per_year).
Метрики по окну (mask) считаются только по барам окна: equity окна стартует
с 1, CAGR и MDD — по ней. Издержки баров окна учтены в net движка;
состояние на входе в окно (позиция, перенесённая из прошлого) уже отражено
в held/turnover. Поэтому metrics(bt, mask) = metrics(run(вырезанный ряд,
initial_pos, prev_close)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _window(bt: pd.DataFrame, mask: pd.Series | np.ndarray | None) -> pd.DataFrame:
    if mask is None:
        return bt
    m = np.asarray(mask, dtype=bool)
    if len(m) != len(bt):
        raise ValueError("длина mask не совпадает с bt")
    pos = np.flatnonzero(m)
    if len(pos) and (pos[-1] - pos[0] + 1 != len(pos)):
        raise ValueError("mask должна быть непрерывным окном")
    return bt.iloc[pos]


def trades(bt: pd.DataFrame) -> pd.DataFrame:
    """Сделки как отрезки постоянного ненулевого знака удерживаемой позиции.

    Переворот закрывает одну сделку и открывает другую. PnL сделки —
    аддитивно: Σ (gross − funding) за бары удержания минус её доля издержек
    на оборот. Издержки бара t, где pos меняется, делятся пропорционально:
    |pos_{t−1}| — закрываемой (если знак сменился или позиция обнулена),
    |pos_t| — открываемой; изменение размера без смены знака целиком
    относится к текущей сделке. Доли, попадающие на сделку вне окна, отброшены.

    Колонки: entry (первый бар удержания), exit (последний), side, bars, pnl.
    """
    if bt.empty:
        return pd.DataFrame(columns=["entry", "exit", "side", "bars", "pnl"])
    held = bt["held"].to_numpy()
    pos = bt["pos"].to_numpy()
    sgn = np.sign(held)
    # id сделки на барах удержания
    start = (sgn != 0) & (np.r_[0.0, sgn[:-1]] != sgn)
    tid = np.where(sgn != 0, np.cumsum(start), 0)
    n_tr = int(tid.max()) if len(tid) else 0
    if n_tr == 0:
        return pd.DataFrame(columns=["entry", "exit", "side", "bars", "pnl"])
    pnl = np.zeros(n_tr + 1)
    np.add.at(pnl, tid, (bt["gross"] - bt["funding"]).to_numpy())
    cost = (bt["fee"] + bt["slippage"]).to_numpy()
    # held следующего бара = pos текущего; для последнего бара сделки — нет
    nxt_tid = np.r_[tid[1:], 0]
    prev_pos = held                                   # pos_{t−1}
    same = (np.sign(prev_pos) == np.sign(pos)) & (pos != 0)
    a_old, a_new = np.abs(prev_pos), np.abs(pos)
    denom = np.where(a_old + a_new > 0, a_old + a_new, 1.0)
    old_share = np.where(same, 0.0, a_old / denom)
    new_share = np.where(same, 1.0, a_new / denom)
    # при same: сделка бара t продолжается на t+1 → вся доля на nxt_tid (= tid)
    np.add.at(pnl, tid, -cost * old_share)
    np.add.at(pnl, nxt_tid, -cost * new_share)
    idx = bt.index
    ids = np.arange(1, n_tr + 1)
    first = np.array([np.flatnonzero(tid == i)[0] for i in ids])
    last = np.array([np.flatnonzero(tid == i)[-1] for i in ids])
    return pd.DataFrame({"entry": idx[first], "exit": idx[last],
                         "side": sgn[first].astype(int), "bars": last - first + 1,
                         "pnl": pnl[1:]})


def max_drawdown(equity: np.ndarray) -> float:
    """Максимальная просадка equity (≤ 0) с учётом стартового капитала 1."""
    e = np.r_[1.0, equity]
    peak = np.maximum.accumulate(e)
    return float((e / peak - 1.0).min())


def _log_equity(net: np.ndarray) -> np.ndarray | None:
    """log equity окна; None — капитал обнулён (net ≤ −1)."""
    if (net <= -1.0).any():
        return None
    return np.cumsum(np.log1p(net))


def _mdd_log(loge: np.ndarray) -> float:
    """MDD по log equity: устойчиво к переполнению на длинных рядах."""
    le = np.r_[0.0, loge]
    return float(np.expm1((le - np.maximum.accumulate(le)).min()))


def metrics(bt: pd.DataFrame, *, periods_per_year: float,
            mask: pd.Series | np.ndarray | None = None) -> dict[str, float]:
    """Метрики по результату engine.run (опционально на непрерывном окне mask).

    - ann_return: CAGR по equity окна (старт с 1): eq_end^(P/n) − 1
      (в логарифмах — без переполнения на длинных рядах);
    - ann_vol: std(net) · √P (ddof=1);
    - sharpe: mean(net) / std(net) · √P;
    - mdd: максимальная просадка equity окна;
    - exposure: доля баров с held ≠ 0;
    - win_rate: доля сделок с pnl > 0 (сделки, пересекающие окно, — по их части в окне);
    - trades_per_year, turnover_per_year (Σ|Δpos| в год);
    - open_at_end: позиция на закрытии последнего бара ≠ 0;
    - n_bars, n_trades, total_cost (Σ fee + slippage + funding).
    """
    w = _window(bt, mask)
    n = len(w)
    if n == 0:
        raise ValueError("пустое окно")
    p = float(periods_per_year)
    net = w["net"].to_numpy()
    loge = _log_equity(net)
    years = n / p
    sd = net.std(ddof=1) if n > 1 else np.nan
    tr = trades(w)
    return {
        "ann_return": float(np.expm1(loge[-1] / years)) if loge is not None else -1.0,
        "ann_vol": float(sd * np.sqrt(p)),
        "sharpe": float(net.mean() / sd * np.sqrt(p)) if sd and sd > 0 else np.nan,
        "mdd": _mdd_log(loge) if loge is not None else -1.0,
        "exposure": float((w["held"] != 0).mean()),
        "win_rate": float((tr["pnl"] > 0).mean()) if len(tr) else np.nan,
        "trades_per_year": len(tr) / years,
        "turnover_per_year": float(w["turnover"].sum() / years),
        "open_at_end": bool(w["pos"].iloc[-1] != 0),
        "n_bars": n,
        "n_trades": len(tr),
        "total_cost": float((w["fee"] + w["slippage"] + w["funding"]).sum()),
    }
