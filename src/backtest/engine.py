"""Векторный бэктест одной позиции на одной паре.

Сигнал считается по закрытию бара t, позиция действует с бара t+1:
    net_t = held_t · ret_t − fee_t − slippage_t − funding_t,  held_t = pos_{t−1},
где ret_t = close_t / close_{t−1} − 1 (простая доходность — аддитивна по
активам при агрегации портфеля); на первом баре held = initial_pos, ret = close_0/prev_close − 1 (без
prev_close — 0).
fee_t = fee_per_side · |Δpos_t|, slippage_t = slippage_per_side · |Δpos_t|.

Допущение: pos — доля капитала, держится постоянной на каждом баре
(equity = Π(1 + net)). Для лонга 1.0 это совпадает с удержанием контрактов;
для шорта и дробных позиций — неявная ребалансировка, комиссией не облагается.
Исполнение — по close t, проскальзывание задаётся явно (по умолчанию 0).

Позиция ∈ [−1, 1], дробная допустима (ансамбль). Хранится полный ряд,
включая нулевые бары. Бары до active_from (usable_from пары) обнуляются здесь
и только здесь: позиция из прогрева в рабочий период не переносится, вход
на первом рабочем баре платит комиссию.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import costs

COLUMNS = ["pos", "held", "ret", "gross", "turnover", "fee", "slippage",
           "funding", "net", "equity"]


def run(df: pd.DataFrame, pos: pd.Series, *, fee_per_side: float,
        slippage_per_side: float = 0.0, funding: pd.Series | None = None,
        tf: str | None = None, initial_pos: float = 0.0,
        prev_close: float | None = None,
        active_from: pd.Timestamp | None = None) -> pd.DataFrame:
    """Прогнать позицию по ценам.

    Args:
        df: свечи, индекс — время открытия бара, колонка close обязательна.
        pos: целевая позиция на закрытии бара t, тот же индекс.
        fee_per_side: комиссия, доля оборота за сторону.
        slippage_per_side: надбавка к комиссии на оборот.
        funding: ставки по моментам начисления; None — без funding.
        tf: таймфрейм (нужен для привязки funding к барам).
        initial_pos: позиция до первого бара (состояние на входе в окно).
        prev_close: close бара перед первым — доходность первого бара;
            вместе с initial_pos задаёт входное состояние окна.
        active_from: бары с open < active_from получают pos = 0; если окно
            начинается в прогреве, initial_pos тоже обнуляется.

    Returns:
        DataFrame на индексе df с колонками COLUMNS.

    Raises:
        ValueError: |pos| > 1, NaN в pos, несовпадение индексов, нет tf при funding.
    """
    if not pos.index.equals(df.index):
        raise ValueError("индекс pos не совпадает с индексом df")
    if pos.isna().any():
        raise ValueError("NaN в позиции")
    if (pos.abs() > 1 + 1e-12).any() or abs(initial_pos) > 1 + 1e-12:
        raise ValueError("|pos| > 1")
    if funding is not None and len(funding) and tf is None:
        raise ValueError("для funding нужен tf")

    pos = pos.astype("float64").copy()
    if active_from is not None:
        warm = df.index < active_from
        pos[warm] = 0.0
        if len(df) and warm[0]:
            initial_pos = 0.0

    close = df["close"].astype("float64")
    ret = close.pct_change()
    ret.iloc[:1] = (close.iloc[0] / prev_close - 1.0) if (prev_close and len(close)) else 0.0
    held = costs.held_position(pos, initial_pos)
    gross = held * ret
    to = costs.turnover(pos, initial_pos)
    fee = fee_per_side * to
    slip = slippage_per_side * to
    fund = (costs.funding_cost(pos, funding, df.index, tf, initial_pos)
            if funding is not None and len(funding) else pd.Series(0.0, index=df.index))
    net = gross - fee - slip - fund
    out = pd.DataFrame({"pos": pos, "held": held, "ret": ret, "gross": gross,
                        "turnover": to, "fee": fee, "slippage": slip,
                        "funding": fund, "net": net}, index=df.index)
    out["equity"] = np.cumprod(1.0 + out["net"].to_numpy())
    return out


def run_for(df: pd.DataFrame, pos: pd.Series, *, symbol: str, tf: str, cfg: dict,
            funding: pd.Series | None = None, initial_pos: float = 0.0,
            prev_close: float | None = None) -> pd.DataFrame:
    """run() с издержками из config.yaml и active_from = usable_from пары.

    Funding применяется, если costs.include_funding.
    """
    from src.data.universe import usable_from

    c = cfg["costs"]
    return run(df, pos, fee_per_side=float(c["fee_per_side"]),
               slippage_per_side=float(c.get("slippage_per_side", 0.0)),
               funding=funding if c.get("include_funding", True) else None,
               tf=tf, initial_pos=initial_pos, prev_close=prev_close,
               active_from=usable_from(cfg, symbol))
