"""S4: Supertrend на ATR Уайлдера (тренд).

TR и ATR Уайлдера — по рабочим барам (_common.true_range, atr_wilder).
basic_up = hl2 + m·ATR_n, basic_dn = hl2 − m·ATR_n, hl2 = (H+L)/2.
Финальные полосы (храповик):
  up_t = basic_up_t, если basic_up_t < up_{t−1} или close_{t−1} > up_{t−1}, иначе up_{t−1};
  dn_t = basic_dn_t, если basic_dn_t > dn_{t−1} или close_{t−1} < dn_{t−1}, иначе dn_{t−1}.
Направление: при dir_{t−1} = +1 → dir_t = −1, если close_t < dn_t;
             при dir_{t−1} = −1 → dir_t = +1, если close_t > up_t.
Начальное dir на первом баре с ATR: +1, если close ≥ hl2, иначе −1 (условность,
влияет только на момент первого разворота). Вся рекурсия — по ряду рабочих
баров; t−1 — предыдущий рабочий бар.

Машина состояний (решение на close t):
- вход long:  разворот dir с −1 на +1 на баре t (событие);
  вход short: разворот с +1 на −1;
- выход — обратный разворот; он же встречный сигнал → переворот на том же баре;
- развороты до конца прогрева (3n рабочих баров) не исполняются: на конце
  прогрева в рынок посреди тренда не входим, ждём первого разворота;
- бар простоя — состояние заморожено.
После первого входа позиция = dir (стоп-и-переворот), экспозиция ≈ 100%.

Ловушки:
- look-ahead: рекурсия использует close_{t−1} и H/L/C бара t — известно на close t;
- прогрев 3n: остаточный вес начального ATR (1 − 1/n)^{3n} ≈ e^{−3} ≈ 5%;
- зависимость от начала ряда: рекурсия с состоянием — сигнал по df[k:] не
  обязан совпадать с сигналом по полному ряду бар в бар (это не look-ahead).
  Поэтому сигнал считается по всей истории, окно вырезается после
  (правило CLAUDE.md, «Подход A»);
- whipsaw: малый m — частые развороты; большой m — поздние входы.
Отличие от автора: у него ATR = SMA(TR), не Уайлдер.

Ex-ante показатель — cost_to_band: издержки круга к m·ATR/close (типичная
дистанция до разворота). С cost_to_target (S1) и cost_to_width (S3) не смешивать.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies._common import atr_wilder, downtime_mask, run_state_machine

WARMUP_PARAMS = ("n",)
COST_METRIC = ("median_band", "cost_to_band")


def _validate(n: int, m: float) -> None:
    if int(n) != n or n < 2:
        raise ValueError("n — целое ≥ 2")
    if not m > 0:
        raise ValueError("m > 0")


def warmup(*, n: int, **_) -> int:
    """Баров прогрева без простоя: 3n."""
    return 3 * int(n)


def supertrend(df: pd.DataFrame, *, n: int, m: float) -> pd.DataFrame:
    """atr, up, dn, dir, line на индексе df; NaN до ATR и на барах простоя."""
    _validate(n, m)
    down = downtime_mask(df)
    w = df.loc[~down, ["high", "low", "close"]].astype("float64")
    atr = atr_wilder(df, n)[~down].to_numpy()
    hl2 = ((w["high"] + w["low"]) / 2).to_numpy()
    c = w["close"].to_numpy()
    k = len(c)
    up = np.full(k, np.nan)
    dn = np.full(k, np.nan)
    d = np.full(k, np.nan)
    valid = np.flatnonzero(~np.isnan(atr))
    if len(valid):
        s = valid[0]
        bu, bd = hl2 + m * atr, hl2 - m * atr
        up[s], dn[s] = bu[s], bd[s]
        d[s] = 1.0 if c[s] >= hl2[s] else -1.0
        for t in range(s + 1, k):
            up[t] = bu[t] if (bu[t] < up[t - 1] or c[t - 1] > up[t - 1]) else up[t - 1]
            dn[t] = bd[t] if (bd[t] > dn[t - 1] or c[t - 1] < dn[t - 1]) else dn[t - 1]
            if d[t - 1] > 0:
                d[t] = -1.0 if c[t] < dn[t] else 1.0
            else:
                d[t] = 1.0 if c[t] > up[t] else -1.0
    line = np.where(d > 0, dn, np.where(d < 0, up, np.nan))
    out = pd.DataFrame({"atr": atr, "up": up, "dn": dn, "dir": d, "line": line},
                       index=w.index)
    return out.reindex(df.index)


def target_move(df: pd.DataFrame, *, n: int, m: float) -> pd.Series:
    """m·ATR_n / close — для cost_to_band."""
    _validate(n, m)
    return m * atr_wilder(df, n) / df["close"].astype("float64")


def signal(df: pd.DataFrame, *, n: int, m: float) -> pd.Series:
    """Позиция ∈ {−1, 0, 1} на индексе df; на прогреве 0."""
    st = supertrend(df, n=n, m=m)
    down = downtime_mask(df)
    d = st["dir"].to_numpy()[~down]                  # ряд рабочих баров
    prev = np.r_[np.nan, d[:-1]]
    flip_up = (d > 0) & (prev < 0)
    flip_dn = (d < 0) & (prev > 0)
    wu = warmup(n=n)
    flip_up[:wu] = False
    flip_dn[:wu] = False
    el = np.zeros(len(df), bool)
    es = np.zeros(len(df), bool)
    el[~down], es[~down] = flip_up, flip_dn
    pos = run_state_machine(el, es, es, el, frozen=down)
    return pd.Series(pos, index=df.index, name="pos")
