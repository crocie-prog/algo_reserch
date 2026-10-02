"""Издержки: комиссия, проскальзывание и funding.

Соглашение о времени: индекс бара — время открытия t, бар покрывает (t, t+Δ].
Позиция pos_t — решение на close t; удерживаемая на баре t позиция —
held_t = pos_{t−1} (на первом баре — initial_pos).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import tf_delta


def turnover(pos: pd.Series, initial_pos: float = 0.0) -> pd.Series:
    """|pos_t − pos_{t−1}|, pos_{−1} = initial_pos. Переворот +1 → −1 = 2."""
    prev = pos.shift(1)
    prev.iloc[:1] = initial_pos
    return (pos - prev).abs()


def fee_cost(pos: pd.Series, fee_per_side: float, initial_pos: float = 0.0) -> pd.Series:
    """cost_t = fee_per_side · |pos_t − pos_{t−1}|.

    Комиссия относится к бару t, на закрытии которого произошла сделка.
    Проскальзывание считается той же функцией со своей ставкой.
    """
    return fee_per_side * turnover(pos, initial_pos)


def held_position(pos: pd.Series, initial_pos: float = 0.0) -> pd.Series:
    """Позиция, удерживаемая на баре t: pos_{t−1}; на первом баре — initial_pos."""
    held = pos.shift(1)
    held.iloc[:1] = initial_pos
    return held


def funding_cost(pos: pd.Series, funding: pd.Series, bar_index: pd.DatetimeIndex,
                 tf: str, initial_pos: float = 0.0) -> pd.Series:
    """Funding, приписанный барам.

    Начисление в момент T списывается с позиции, удерживаемой до T, то есть
    на баре, который заканчивается в T: open = T − Δ, позиция held = pos.shift(1)
    на этом баре. Если внутри бара несколько начислений (1d: 08:00, 16:00 и
    00:00 следующего дня), они суммируются. Метки вне диапазона баров
    игнорируются.

    Знак: cost = held · rate. Положительная ставка — лонг платит
    (cost > 0), шорт получает (cost < 0).

    Returns:
        Серия на bar_index; 0, где начислений нет.
    """
    delta = tf_delta(tf)
    if funding is None or len(funding) == 0:
        return pd.Series(0.0, index=bar_index)
    t = pd.DatetimeIndex(funding.index)
    # бар, заканчивающийся в T: open = ceil(T) − Δ (T на сетке → T − Δ)
    bar_open = t.ceil(delta) - delta
    rate = pd.Series(np.asarray(funding, dtype="float64"), index=bar_open)
    per_bar = rate.groupby(level=0).sum().reindex(bar_index, fill_value=0.0)
    held = held_position(pos.reindex(bar_index), initial_pos)
    return held * per_bar
