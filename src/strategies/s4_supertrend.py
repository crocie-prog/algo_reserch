"""S4: Supertrend на ATR Уайлдера (тренд).

basic_up = hl2 + m·ATR_n, basic_dn = hl2 − m·ATR_n, hl2 = (H+L)/2.
Рекурсия финальных полос:
  up_t = basic_up_t, если basic_up_t < up_{t−1} или close_{t−1} > up_{t−1}, иначе up_{t−1};
  dn_t = basic_dn_t, если basic_dn_t > dn_{t−1} или close_{t−1} < dn_{t−1}, иначе dn_{t−1}.
Направление: при dir_{t−1} = +1 → dir_t = −1, если close_t < dn_t;
             при dir_{t−1} = −1 → dir_t = +1, если close_t > up_t.

Машина состояний (решение на close t):
- вход long:  разворот dir с −1 на +1 на баре t;
  вход short: разворот с +1 на −1;
- выход long: разворот вниз; выход short: разворот вверх;
- встречный сигнал совпадает с выходом → переворот на том же баре.
Вход по событию разворота, а не по уровню: на конце прогрева в рынок
посреди тренда не входим, ждём первого разворота.

Прогрев: ATR Уайлдера сходится примерно за 3·n баров → warmup = 3·n
(и не меньше первого определённого dir).
"""
from __future__ import annotations

import pandas as pd

WARMUP_PARAMS = ("n",)


def warmup(*, n: int, **_) -> int:
    """Баров прогрева: 3·n."""
    raise NotImplementedError


def signal(df: pd.DataFrame, *, n: int, m: float) -> pd.Series:
    """Позиция ∈ {−1, 0, 1} на индексе df; на прогреве 0."""
    raise NotImplementedError
