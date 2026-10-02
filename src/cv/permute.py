"""Перестановочные данные для базовой линии шума (диагностика, не попытка).

Перемешиваются целые сутки UTC (полные дни; неполные первый и последний
остаются на месте). Каждый бар хранит относительные величины к close
предыдущего бара: c/c₋₁, o/c₋₁, h/c₋₁, l/c₋₁, объём, turnover/(volume·c),
is_downtime. После перестановки цены восстанавливаются цепочкой от
исходного стартового close, метки времени не меняются. Внутридневная
структура сохраняется (консервативно для S5/S6), межсуточная — разрушается.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def permute_days(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Копия df с перемешанными полными сутками."""
    c = df["close"].to_numpy(dtype="float64")
    prev = np.r_[c[0], c[:-1]]
    rel = {k: df[k].to_numpy(dtype="float64") / prev for k in ("open", "high", "low", "close")}
    vol = df["volume"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        vw_rel = np.where(vol > 0, df["turnover"].to_numpy(dtype="float64") / (vol * c), 0.0)
    down = df["is_downtime"].to_numpy(bool) if "is_downtime" in df else np.zeros(len(df), bool)

    day = df.index.floor("1D")
    counts = pd.Series(1, index=day).groupby(level=0).size()
    full_len = counts.max()
    days = counts.index.to_numpy()
    full = [d for d in days if counts[d] == full_len]
    first, last = days[0], days[-1]
    movable = [d for d in full if d != first and d != last]
    order = list(rng.permutation(len(movable)))
    mapping = {movable[i]: movable[j] for i, j in enumerate(order)}
    pos_of_day = {d: np.flatnonzero(day == d) for d in days}
    src = np.concatenate([pos_of_day[mapping.get(d, d)] for d in days])

    r_close = rel["close"][src].copy()
    r_close[0] = 1.0
    new_c = c[0] * np.cumprod(r_close)
    new_prev = np.r_[c[0], new_c[:-1]]
    out = pd.DataFrame(index=df.index)
    out["open"] = rel["open"][src] * new_prev
    out["high"] = rel["high"][src] * new_prev
    out["low"] = rel["low"][src] * new_prev
    out["close"] = new_c
    out["volume"] = vol[src]
    out["turnover"] = vol[src] * vw_rel[src] * new_c
    if "is_downtime" in df:
        out["is_downtime"] = down[src]
    return out
