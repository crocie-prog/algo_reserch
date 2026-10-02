"""S3: Donchian Channel (тренд).

HH_t = max(high), LL_t = min(low) по w РАБОЧИМ барам ДО t (бар t не входит):
rolling по ряду рабочих баров со сдвигом на 1 внутри этого ряда, затем
перенос на исходный индекс. На первом рабочем баре после простоя канал —
последние w рабочих баров до простоя.

Машина состояний (решение на close t):
- вход long:  close_t > HH_t;  вход short: close_t < LL_t (строго);
- выход long: close_t < LL_t (противоположная граница канала);
  выход short: close_t > HH_t;
- встречный сигнал совпадает с выходом → переворот на том же баре.
- бар простоя — состояние заморожено.
Следствие: после первого входа стратегия всегда в рынке (стоп-и-переворот),
экспозиция ≈ 100%, почти все входы — перевороты.

Ловушки:
- look-ahead/смещение: если включить бар t в канал, close_t ≤ high_t ≤ HH_t —
  пробоя вверх не бывает; сдвиг не в ту сторону — заглядывание в будущее;
- прогрев — w баров (канал по w барам до t, первое решение на баре w);
- whipsaw — в боковике переворот на каждом пересечении ширины канала,
  при малом w часто;
- равенство границе — не пробой.
Исполнение по close пробоя — вход после движения (для тренда не оптимистично
в отличие от контртренда); чувствительность к slippage — единым прогоном для
всех стратегий на этапе 5.

Ex-ante показатель — cost_to_width: издержки круга к ширине канала
(HH − LL)/close, т.е. к расстоянию до точки разворота. Это не цель сделки,
как у S1; в таблицах с cost_to_target не смешивать.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies._common import downtime_mask, run_state_machine

WARMUP_PARAMS = ("w",)
COST_METRIC = ("median_width", "cost_to_width")


def _validate(w: int) -> None:
    if int(w) != w or w < 2:
        raise ValueError("w — целое ≥ 2")


def warmup(*, w: int, **_) -> int:
    """Баров прогрева без простоя: w (канал по w барам до t)."""
    return int(w)


def channel(df: pd.DataFrame, *, w: int) -> pd.DataFrame:
    """hh, ll по w рабочим барам до t; NaN на прогреве и на барах простоя."""
    _validate(w)
    down = downtime_mask(df)
    work = df.loc[~down, ["high", "low"]].astype("float64")
    hh = work["high"].rolling(w, min_periods=w).max().shift(1)
    ll = work["low"].rolling(w, min_periods=w).min().shift(1)
    return pd.DataFrame({"hh": hh, "ll": ll}).reindex(df.index)


def target_move(df: pd.DataFrame, *, w: int) -> pd.Series:
    """Ширина канала в долях цены: (hh − ll) / close. Для cost_to_width."""
    ch = channel(df, w=w)
    return (ch["hh"] - ch["ll"]) / df["close"].astype("float64")


def signal(df: pd.DataFrame, *, w: int) -> pd.Series:
    """Позиция ∈ {−1, 0, 1} на индексе df; на прогреве 0."""
    ch = channel(df, w=w)
    c = df["close"].to_numpy(dtype="float64")
    hh, ll = ch["hh"].to_numpy(), ch["ll"].to_numpy()
    ok = ~np.isnan(hh) & ~np.isnan(ll)
    up = ok & (c > np.where(ok, hh, np.inf))
    dn = ok & (c < np.where(ok, ll, -np.inf))
    pos = run_state_machine(up, dn, dn, up, frozen=downtime_mask(df))
    return pd.Series(pos, index=df.index, name="pos")
