"""S1: Z-score от скользящей средней (контртренд).

z_t = (close_t − SMA_w,t) / STD_w,t; SMA и STD (ddof=1) — по последним w
РАБОЧИМ барам (бары простоя в окно не входят), включая бар t: решение
принимается на close t, позиция действует с t+1 (сдвиг делает движок).

Машина состояний (решение на close t):
- вход long:  z_t < −z_entry;  вход short: z_t > z_entry;
- выход long: z_t ≥ −z_exit (возврат z к −z_exit);
  выход short: z_t ≤ z_exit;
- встречный сигнал (z за противоположным порогом входа) — выход;
  на том же баре вход в обратную сторону по условию входа (переворот).
- z не определён (прогрев, STD = 0) — состояние не меняется;
- бар простоя — состояние заморожено (ни входа, ни выхода).
Ограничение: 0 ≤ z_exit < z_entry.

Отличие от автора (strategies_walkforward.py, gen_s1_signals): там выход
|z| < z_exit; скачок z через полосу выхода не закрывает позицию, а при
z_exit = 0 выход по сигналу невозможен. Здесь выход — по односторонней
границе. Фильтры автора (свечи, сила сигнала, режим волатильности,
объём) не переносятся.

Ловушки: whipsaw — узкая полоса гистерезиса [−z_entry, −z_exit) даёт частые
входы и выходы около порога; при z_exit ≈ 0 выход ровно на средней и быстрый
повторный вход. Прогрев — w − 1 баров (z определён с w-го рабочего бара).

Допущение исполнения: сделка по цене сигнального close. Для контртрендовой
стратегии это оптимистично — вход происходит на экстремуме отклонения,
откуда цена чаще всего уже отходит. На этапе 5 обязателен прогон
чувствительности к slippage_per_side: 0 / 0.05% / 0.1%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies._common import downtime_mask, rolling_std, run_state_machine, sma

WARMUP_PARAMS = ("w",)


def _validate(w: int, z_entry: float, z_exit: float) -> None:
    if int(w) != w or w < 2:
        raise ValueError("w — целое ≥ 2")
    if not (0 <= z_exit < z_entry):
        raise ValueError("нужно 0 ≤ z_exit < z_entry")


def warmup(*, w: int, **_) -> int:
    """Баров прогрева без простоя: w − 1 (z определён с w-го рабочего бара)."""
    return int(w) - 1


def zscore(df: pd.DataFrame, *, w: int) -> pd.Series:
    """z по рабочим барам; NaN на прогреве, на барах простоя и при STD = 0."""
    down = downtime_mask(df)
    close = df["close"].astype("float64")
    mu = sma(close, w, down)
    sd = rolling_std(close, w, down)
    z = (close - mu) / sd.where(sd > 0)
    z[down] = np.nan
    return z


def target_move(df: pd.DataFrame, *, w: int, z_entry: float, z_exit: float) -> pd.Series:
    """Ex-ante цель сделки в долях цены: (z_entry − z_exit) · STD_w / close.

    Путь z от порога входа до порога выхода в единицах σ окна. Используется
    только для описательного показателя cost_to_target; доходность не считается.
    NaN на прогреве и барах простоя.
    """
    _validate(w, z_entry, z_exit)
    down = downtime_mask(df)
    close = df["close"].astype("float64")
    sd = rolling_std(close, w, down)
    out = (z_entry - z_exit) * sd / close
    out[down] = np.nan
    return out


def signal(df: pd.DataFrame, *, w: int, z_entry: float, z_exit: float) -> pd.Series:
    """Позиция ∈ {−1, 0, 1} на индексе df; на прогреве 0."""
    _validate(w, z_entry, z_exit)
    z = zscore(df, w=w).to_numpy()
    ok = ~np.isnan(z)
    zz = np.where(ok, z, 0.0)
    el = ok & (zz < -z_entry)
    es = ok & (zz > z_entry)
    xl = ok & (zz >= -z_exit)
    xs = ok & (zz <= z_exit)
    pos = run_state_machine(el, es, xl, xs, frozen=downtime_mask(df))
    return pd.Series(pos, index=df.index, name="pos")
