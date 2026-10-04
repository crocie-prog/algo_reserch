"""Фильтр направления по долгосрочной средней (гипотеза H4).

SMA = простая средняя close по `sma_bars` рабочим барам (бары простоя не
входят в окно; на 4h 200 дней = 1200 баров, config strategies.sma_filter).
Лонг разрешён при close_t > SMA_t, шорт — при close_t < SMA_t; при
равенстве и на прогреве (SMA ещё нет) оба направления запрещены.
На барах простоя разрешения заморожены (переносятся с последнего рабочего
бара), как состояние стратегий.

Ловушки:
- look-ahead: SMA_t и close_t известны на закрытии t; позиция — с t+1 (движок);
- прогрев: sma_bars рабочих баров — позиция 0;
- whipsaw около SMA: гистерезиса нет (это был бы ещё один параметр).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies._common import downtime_mask, sma as rolling_sma


def config_params(tf: str, cfg: dict) -> dict:
    """Длина средней из config (не параметр сетки)."""
    return {"sma_bars": int(cfg["strategies"]["sma_filter"]["bars"][tf])}


def allowed(df: pd.DataFrame, *, sma_bars: int) -> pd.DataFrame:
    """Колонки long, short (bool) и sma на индексе df."""
    down = downtime_mask(df)
    close = df["close"].astype("float64")
    sma = rolling_sma(close, sma_bars, down)
    lo = pd.Series(np.where(sma.isna(), np.nan, (close > sma).astype(float)), index=df.index)
    sh = pd.Series(np.where(sma.isna(), np.nan, (close < sma).astype(float)), index=df.index)
    if down.any():                                    # простой: заморозка разрешений
        lo[down] = np.nan
        sh[down] = np.nan
        lo, sh = lo.ffill(), sh.ffill()
    return pd.DataFrame({"long": lo.fillna(0).astype(bool), "short": sh.fillna(0).astype(bool),
                         "sma": sma}, index=df.index)


def apply(pos: pd.Series, allow: pd.DataFrame) -> pd.Series:
    """Маска: позиция сохраняется, если её знак разрешён, иначе 0."""
    p = pos.to_numpy(dtype="float64")
    keep = ((p > 0) & allow["long"].to_numpy()) | ((p < 0) & allow["short"].to_numpy())
    return pd.Series(np.where(keep, p, 0.0), index=pos.index, name="pos")
