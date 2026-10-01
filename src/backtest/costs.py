"""Издержки: комиссия и funding.

Соглашение о времени: индекс бара — время открытия t, бар покрывает (t, t+Δ].
Позиция, удерживаемая на баре t, — pos.shift(1)[t] (решение на close t−1).
"""
from __future__ import annotations

import pandas as pd


def fee_cost(pos: pd.Series, fee_per_side: float) -> pd.Series:
    """cost_t = fee_per_side · |pos_t − pos_{t−1}|, pos_{−1} = 0.

    Переворот +1 → −1 даёт |Δpos| = 2, то есть два оборота.
    Комиссия относится к бару t, на закрытии которого произошла сделка.
    """
    raise NotImplementedError


def funding_cost(pos: pd.Series, funding: pd.Series, bar_index: pd.DatetimeIndex,
                 tf: str) -> pd.Series:
    """Funding, приписанный барам.

    Начисление в момент T списывается с позиции, удерживаемой до T, то есть
    на баре, который заканчивается в T: open = T − Δ, позиция pos.shift(1)
    на этом баре. Если внутри бара несколько начислений (1d: 08:00, 16:00 и
    00:00 следующего дня), они суммируются.

    Знак: cost = held_pos · rate. Положительная ставка — лонг платит
    (cost > 0), шорт получает (cost < 0).

    Returns:
        Серия на bar_index; 0, где начислений нет.
    """
    raise NotImplementedError
