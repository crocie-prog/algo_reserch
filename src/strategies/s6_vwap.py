"""S6: VWAP ± отклонение (диапазон).

Якорь задаётся в config.yaml (strategies.s6.anchor), не параметр сетки:
- session (1h, 15m): накопление с 00:00 UTC текущего дня,
  VWAP_t = Σ(tp·volume)/Σvolume по барам сессии до t включительно,
  tp = (H+L+C)/3, volume — base;
  dev_t = std(close − VWAP) по барам сессии до t (ddof=1);
- rolling (1d): те же суммы по окну w баров, dev — rolling std(close − VWAP, w).

Машина состояний (решение на close t):
- вход long:  close_t < VWAP_t − k·dev_t;  вход short: close_t > VWAP_t + k·dev_t;
- выход long/short: |close_t − VWAP_t| ≤ e·dev_t (возврат к VWAP), 0 ≤ e < k;
- встречный сигнал (close за противоположной полосой) — выход и переворот.

Ловушки: в начале сессии dev ≈ 0 → почти нулевой порог входа, whipsaw;
входы разрешены только после min_session_bars баров сессии. Позиция
переносится через 00:00 UTC, выход после смены дня — по VWAP новой сессии.
Прогрев: rolling — w баров; session — до первой полной сессии.
"""
from __future__ import annotations

import pandas as pd

WARMUP_PARAMS = ("w", "min_session_bars")


def warmup(*, anchor: str, w: int | None = None,
           min_session_bars: int | None = None, **_) -> int:
    """Баров прогрева для заданного якоря."""
    raise NotImplementedError


def vwap(df: pd.DataFrame, *, anchor: str, w: int | None = None) -> pd.DataFrame:
    """VWAP и dev на индексе df."""
    raise NotImplementedError


def signal(df: pd.DataFrame, *, anchor: str, k: float, e: float,
           w: int | None = None, min_session_bars: int | None = None) -> pd.Series:
    """Позиция ∈ {−1, 0, 1} на индексе df; на прогреве 0."""
    raise NotImplementedError
