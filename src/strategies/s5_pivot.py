"""S5: Pivot Points S1/R1 (диапазон), без параметров.

Уровни дня D (UTC, граница 00:00) — только по ЗАКРЫТОМУ дню D−1:
  P = (H + L + C)/3,  S1 = 2P − H,  R1 = 2P − L  (H, L, C дня D−1).
День D−1 агрегируется из баров рабочего ТФ (H = max, L = min, C = последний
close); сутки должны быть полными — 1440 / минуты ТФ баров, бары простоя
считаются (в ряду они есть как плоские и не искажают H и L). Clean 15m/1h —
агрегат 1m, поэтому дневной бар совпадает с clean 1d. Неполный или
отсутствующий день D−1 — уровни дня D = NaN, действий нет.

Машина состояний (решение на close t):
- вход long:  close_t < S1_D;  вход short: close_t > R1_D;
- выход long: close_t ≥ P_D (возврат к P); выход short: close_t ≤ P_D;
- встречный сигнал (close за противоположным уровнем) — выход и переворот;
- бар простоя — состояние заморожено.
Позиция переносится через 00:00 UTC; после смены дня выход проверяется по
уровням нового дня. Закрытие в конце дня — отдельная гипотеза.

Ловушки:
- смещение баров: бар 23:00 дня D использует уровни D (HLC дня D−1); бар 00:00
  дня D+1 — уровни D+1 (HLC дня D, закрытого в 00:00; решение на close
  бара 00:00 — look-ahead нет);
- граница дня: уровни меняются скачком — позиция может закрыться или
  открыться на первом баре новых суток (механизм того же рода, что сброс
  VWAP у S6);
- прогрев: до первого бара второго полного дня (нижняя граница — сутки баров,
  если ряд начинается в 00:00);
- whipsaw: колебания close около S1/R1, узкие уровни после тихого дня;
- исполнение по close оптимистично для контртренда; чувствительность к
  slippage — единым прогоном на этапе 5.

Параметров нет (вариант (а)): подход A с топ-10 и картами плато к S5
неприменим; в walk-forward — только правило отказа по порогу train-Sharpe,
в журнал DSR — одна попытка. Варианты автора (тип пивота, уровень 1/2/3,
pivot_frac) — будущая гипотеза; при реализации все уникальные комбинации
идут в журнал DSR.

Ex-ante показатель — cost_to_target: издержки круга к (R1 − S1)/2/close =
(H − L)_{D−1}/2/close (путь от уровня входа до P); сопоставим с S1 и S6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import tf_minutes
from src.strategies._common import downtime_mask, run_state_machine

WARMUP_PARAMS = ()


def config_params(tf: str, cfg: dict) -> dict:
    """tf — для проверки полноты суток (не параметр сетки)."""
    if cfg["strategies"]["s5"].get("day_boundary", "00:00") != "00:00":
        raise ValueError("поддерживается только граница дня 00:00 UTC")
    return {"tf": tf}


def carry_kwargs(tf: str, cfg: dict, params: dict) -> dict | None:
    """Для описательной статистики переноса через 00:00."""
    eb = cfg["strategies"]["s5"].get("carry_early_bars", {}).get(tf)
    return {"early_bars": int(eb)} if eb else None


def bars_per_day(tf: str) -> int:
    m = tf_minutes(tf)
    if 1440 % m:
        raise ValueError(f"ТФ {tf} не делит сутки")
    return 1440 // m


def warmup(*, tf: str, **_) -> int:
    """Нижняя граница прогрева: сутки баров (если ряд начинается в 00:00)."""
    return bars_per_day(tf)


def daily_levels(df: pd.DataFrame, *, tf: str) -> pd.DataFrame:
    """P, S1, R1 на каждом баре по закрытому полному дню D−1; иначе NaN."""
    day = df.index.floor("1D")
    g = df.groupby(day)
    agg = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "n": g["close"].size()})
    agg = agg[agg["n"] == bars_per_day(tf)]
    p = (agg["high"] + agg["low"] + agg["close"]) / 3
    lv = pd.DataFrame({"P": p, "S1": 2 * p - agg["high"], "R1": 2 * p - agg["low"]})
    lv.index = lv.index + pd.Timedelta(days=1)       # уровни дня D — по дню D−1
    out = lv.reindex(day)
    out.index = df.index
    return out


def target_move(df: pd.DataFrame, *, tf: str) -> pd.Series:
    """(R1 − S1)/2/close — для cost_to_target."""
    lv = daily_levels(df, tf=tf)
    return (lv["R1"] - lv["S1"]) / 2 / df["close"].astype("float64")


def signal(df: pd.DataFrame, *, tf: str) -> pd.Series:
    """Позиция ∈ {−1, 0, 1} на индексе df; без уровней — 0."""
    lv = daily_levels(df, tf=tf)
    c = df["close"].to_numpy(dtype="float64")
    p, s1, r1 = (lv[k].to_numpy() for k in ("P", "S1", "R1"))
    ok = ~np.isnan(p)
    p0, s0, r0 = (np.where(ok, x, 0.0) for x in (p, s1, r1))
    el = ok & (c < s0)
    es = ok & (c > r0)
    xl = ok & (c >= p0)
    xs = ok & (c <= p0)
    pos = run_state_machine(el, es, xl, xs, frozen=downtime_mask(df))
    return pd.Series(pos, index=df.index, name="pos")
