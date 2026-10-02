"""S2: Bollinger Bands (контртренд) — тонкая обёртка над S1.

mid = SMA_w, upper/lower = mid ± k·STD_w (ddof=1, по рабочим барам).
Вход long: close ≤ lower ⇔ z ≤ −k; выход long: close ≥ mid ⇔ z ≥ 0
(симметрично для short). Значит, S2(w, k) ≡ S1(w, z_entry=k, z_exit=0);
отличие только в нестрогом неравенстве на границе (совпадение цены
с полосой практически не встречается: на BTC/ETH/SOL 1h train при
w ∈ {24, 72, 168}, k = 2 — 0 отличающихся баров). Поэтому signal()
вызывает s1.signal, а не дублирует логику.

Статус (config.yaml, strategies.s2): S2 не оптимизируется отдельно,
не входит в МСП, попытки не пишутся в журнал DSR — это подмножество
сетки S1. Эффективных стратегий пять: S1, S3, S4, S5, S6.

У автора S1 и S2 различались из-за фильтров (свечи, сила сигнала, режим
волатильности, объём) и выхода S1 по |z| < z_exit; при унифицированных
правилах стратегии совпадают.

Ловушки — как у S1 (look-ahead нет: полосы включают close_t, решение на
close t; прогрев w − 1; простой — заморозка и окна по рабочим барам;
whipsaw — выход на SMA и повторный вход у полосы). Исполнение по close
оптимистично для контртренда — см. S1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies import s1_zscore as s1
from src.strategies._common import downtime_mask, rolling_std, sma

WARMUP_PARAMS = ("w",)


def _validate(w: int, k: float) -> None:
    if not k > 0:
        raise ValueError("k > 0")
    s1._validate(w, k, 0.0)


def warmup(*, w: int, **_) -> int:
    """Баров прогрева без простоя: w − 1."""
    return s1.warmup(w=w)


def bands(df: pd.DataFrame, *, w: int, k: float) -> pd.DataFrame:
    """mid, upper, lower по рабочим барам (для визуализации); NaN на простое."""
    _validate(w, k)
    down = downtime_mask(df)
    close = df["close"].astype("float64")
    mid = sma(close, w, down)
    sd = rolling_std(close, w, down)
    out = pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd})
    out[down] = np.nan
    return out


def target_move(df: pd.DataFrame, *, w: int, k: float) -> pd.Series:
    """Ex-ante цель сделки в долях цены: k · STD_w / close (путь от полосы до SMA)."""
    _validate(w, k)
    return s1.target_move(df, w=w, z_entry=k, z_exit=0.0)


def signal(df: pd.DataFrame, *, w: int, k: float) -> pd.Series:
    """Позиция ∈ {−1, 0, 1}: s1.signal(w, z_entry=k, z_exit=0)."""
    _validate(w, k)
    return s1.signal(df, w=w, z_entry=k, z_exit=0.0)
