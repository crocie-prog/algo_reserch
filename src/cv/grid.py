"""Кэш сетки конфигураций одной стратегии на одной паре × ТФ.

Для каждой конфигурации ОДИН раз по всей истории (до train_end этапа):
- позиция (решение на close; сигналы каузальны — truncation_check этапа 3);
- чистая доходность движка (комиссия, slippage прогона, funding;
  позиции до usable_from обнуляет движок);
- ex-ante цель сделки target_move (для отсечения по издержкам).
Статистики любого окна (train фолда, валидация) — срезы этих матриц.
Это правило CLAUDE.md «рекурсивные стратегии считаются по всей истории,
окно вырезается после» и «метрики по маске» этапа 2.
"""
from __future__ import annotations

import importlib
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest import engine
from src.data.universe import usable_from


@dataclass
class GridData:
    strategy: str
    tf: str
    symbol: str
    params: list[dict]            # полные параметры signal() (сетка + config_params)
    grid_params: list[dict]       # только параметры сетки
    axes: dict[str, list]         # оси сетки (значения по порядку)
    coords: np.ndarray            # (C × n_axes) индексы конфигураций на осях
    index: pd.DatetimeIndex
    pos: np.ndarray               # (T × C) float32
    net: np.ndarray               # (T × C) float64
    target: np.ndarray            # (T × C) float32, NaN где не определено
    round_trip: float             # издержки круга 2·(fee + slippage)
    cost_metric: str              # cost_to_target / cost_to_width / cost_to_band


def expand_grid(axes: dict[str, list]) -> list[dict]:
    """Декартово произведение осей в порядке конфига; пустые оси → [{}]."""
    if not axes:
        return [{}]
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*(axes[k] for k in keys))]


def build_grid(df: pd.DataFrame, funding: pd.Series | None, *, strategy: str, tf: str,
               symbol: str, cfg: dict, slippage: float,
               axes: dict[str, list] | None = None) -> GridData:
    """Посчитать позиции, чистые доходности и цели всех конфигураций сетки."""
    mod = importlib.import_module(f"src.strategies.{strategy}")
    axes = cfg["grids"][strategy][tf] if axes is None else axes
    gp = expand_grid(axes)
    extra = mod.config_params(tf, cfg) if hasattr(mod, "config_params") else {}
    full = [{**extra, **p} for p in gp]
    c = cfg["costs"]
    fee = float(c["fee_per_side"])
    fund = funding if c.get("include_funding", True) else None
    af = usable_from(cfg, symbol)
    n, k = len(df), len(full)
    pos = np.zeros((n, k), dtype="float32")
    net = np.zeros((n, k), dtype="float64")
    tgt = np.full((n, k), np.nan, dtype="float32")
    for j, p in enumerate(full):
        s = mod.signal(df, **p)
        bt = engine.run(df, s, fee_per_side=fee, slippage_per_side=slippage,
                        funding=fund, tf=tf, active_from=af)
        pos[:, j] = bt["pos"].to_numpy()
        net[:, j] = bt["net"].to_numpy()
        if hasattr(mod, "target_move"):
            tgt[:, j] = mod.target_move(df, **p).to_numpy()
    keys = list(axes)
    coords = np.array([[axes[a].index(g[a]) for a in keys] for g in gp], dtype=int) \
        if keys else np.zeros((1, 0), dtype=int)
    metric = getattr(mod, "COST_METRIC", ("median_target", "cost_to_target"))[1]
    return GridData(strategy, tf, symbol, full, gp, dict(axes), coords, df.index,
                    pos, net, tgt, 2.0 * (fee + slippage), metric)


def window_rows(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> slice:
    """Срез строк [start, end) по времени открытия бара."""
    a = int(index.searchsorted(start, side="left"))
    b = int(index.searchsorted(end, side="left"))
    return slice(a, b)


def window_stats(gd: GridData, start: pd.Timestamp, end: pd.Timestamp, *,
                 periods_per_year: float) -> pd.DataFrame:
    """По каждой конфигурации на окне [start, end):

    sharpe (годовой), sr_bar, n_obs, trades_per_year (входы: смена знака,
    состояние на входе в окно — из предыдущего бара), cost_ratio
    (издержки круга / медиана цели на окне; NaN — показателя нет).
    """
    rows = window_rows(gd.index, start, end)
    x = gd.net[rows]
    t = x.shape[0]
    mu = x.mean(axis=0) if t else np.full(x.shape[1], np.nan)
    sd = x.std(axis=0, ddof=1) if t > 1 else np.full(x.shape[1], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mu / sd, np.nan)
    s = np.sign(gd.pos[rows])
    prev_row = np.sign(gd.pos[rows.start - 1]) if rows.start > 0 else np.zeros(s.shape[1])
    prev = np.vstack([prev_row[None, :], s[:-1]]) if t else s
    entries = ((s != 0) & (s != prev)).sum(axis=0)
    years = t / periods_per_year if t else np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(gd.target[rows], axis=0) if t else np.full(s.shape[1], np.nan)
    ratio = np.where(med > 0, gd.round_trip / med, np.nan)
    return pd.DataFrame({"sharpe": sr * np.sqrt(periods_per_year), "sr_bar": sr, "n_obs": t,
                         "trades_per_year": entries / years, "cost_ratio": ratio})


def mask_stats(gd: GridData, rows: np.ndarray, *, periods_per_year: float) -> pd.DataFrame:
    """То же, что window_stats, но по произвольной (в т.ч. несмежной) bool-маске строк.

    Вход считается относительно РЕАЛЬНОГО предыдущего бара ряда (а не
    предыдущего выбранного): позиции непрерывны по всей истории.
    """
    m = np.asarray(rows, dtype=bool)
    idx = np.flatnonzero(m)
    t = len(idx)
    x = gd.net[idx]
    mu = x.mean(axis=0) if t else np.full(x.shape[1], np.nan)
    sd = x.std(axis=0, ddof=1) if t > 1 else np.full(x.shape[1], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mu / sd, np.nan)
    s = np.sign(gd.pos[idx])
    prev_idx = idx - 1
    prev = np.where((prev_idx >= 0)[:, None], np.sign(gd.pos[np.maximum(prev_idx, 0)]), 0.0)
    entries = ((s != 0) & (s != prev)).sum(axis=0)
    years = t / periods_per_year if t else np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(gd.target[idx], axis=0) if t else np.full(s.shape[1], np.nan)
    ratio = np.where(med > 0, gd.round_trip / med, np.nan)
    return pd.DataFrame({"sharpe": sr * np.sqrt(periods_per_year), "sr_bar": sr, "n_obs": t,
                         "trades_per_year": entries / years, "cost_ratio": ratio})
