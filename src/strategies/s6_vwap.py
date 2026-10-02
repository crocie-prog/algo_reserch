"""S6: VWAP ± отклонение (диапазон).

VWAP — истинный: Σ turnover / Σ volume (turnover = Σ цена·объём в quote,
volume — base). У автора — приближение (H+L+C)/3, взвешенное объёмом.

Якорь задаётся в config.yaml (strategies.s6.anchor), не параметр сетки:
- session (1h, 15m): суммы накапливаются по рабочим барам текущих суток UTC
  (сессия бара = floor(время открытия, 1 день)) до бара t включительно;
  dev_t = std(close − VWAP) по рабочим барам сессии до t (ddof=1);
- rolling (1d, только описательно): суммы по w рабочим барам,
  dev — rolling std(close − VWAP, w).

Машина состояний (решение на close t):
- вход long:  close_t < VWAP_t − k·dev_t;  вход short: close_t > VWAP_t + k·dev_t;
- выход long/short: |close_t − VWAP_t| ≤ e·dev_t (возврат к VWAP), 0 ≤ e < k;
- встречный сигнал (close за противоположной полосой) — выход всегда;
  вход в обратную сторону на том же баре — если вход разрешён (см. ниже);
- VWAP или dev не определены (Σ volume = 0, первый бар сессии, прогрев) —
  ничего не меняется; бар простоя — состояние заморожено.

Прогрев сессии: входы разрешены начиная с min_session_bars-го рабочего бара
сессии. min_session_bars — фиксированное значение из config.yaml
(strategies.s6.min_session_bars, по ТФ), НЕ параметр сетки и не
оптимизируется; выходы в начале сессии проверяются как обычно. Если данные
начинаются не в 00:00, первая (неполная) сессия входов не даёт.

Ловушки:
- look-ahead: VWAP и dev включают бар t, решение на close t — допустимо;
  сессия — по времени ОТКРЫТИЯ бара (бар 23:00 — день D, бар 00:00 — D+1);
- граница дня: в 00:00 VWAP и dev обнуляются; позиция переносится через
  полночь, выход проверяется по VWAP новой сессии; на втором баре сессии dev
  по двум точкам ≈ 0 — выход в первые часы малопредсказуем (по спецификации);
- простой: нулевой объём не вносит вклад в VWAP; бары простоя не входят
  в dev и в счёт баров сессии; состояние заморожено;
- whipsaw: узкие полосы в начале сессии (частично закрыто min_session_bars),
  e близко к k — частые входы и выходы.
Исполнение по close оптимистично для контртренда (вход на экстремуме);
чувствительность к slippage и проверка на выборке с 2022 года — этап 5.

Ex-ante показатель — cost_to_target: издержки круга к (k − e)·dev/close
(путь от полосы входа до полосы выхода); сопоставим с cost_to_target S1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies._common import downtime_mask, run_state_machine

WARMUP_PARAMS = ("w",)
ANCHORS = ("session", "rolling")


def config_params(tf: str, cfg: dict) -> dict:
    """anchor и min_session_bars для ТФ из config.yaml (не параметры сетки)."""
    s6 = cfg["strategies"]["s6"]
    anchor = s6["anchor"][tf]
    out = {"anchor": anchor}
    if anchor == "session":
        out["min_session_bars"] = int(s6["min_session_bars"][tf])
    return out


def _validate(anchor: str, k: float, e: float, w: int | None,
              min_session_bars: int | None) -> None:
    if anchor not in ANCHORS:
        raise ValueError(f"anchor ∈ {ANCHORS}")
    if not (0 <= e < k):
        raise ValueError("нужно 0 ≤ e < k")
    if anchor == "rolling" and (w is None or int(w) != w or w < 2):
        raise ValueError("для rolling нужен целый w ≥ 2")
    if anchor == "session" and (min_session_bars is None or int(min_session_bars) != min_session_bars
                                or min_session_bars < 2):
        raise ValueError("для session нужен целый min_session_bars ≥ 2 (из config)")


def warmup(*, anchor: str, w: int | None = None, min_session_bars: int | None = None,
           **_) -> int:
    """Баров прогрева без простоя.

    rolling: 2w − 2 (VWAP по w барам, затем std отклонений по w значениям);
    session: min_session_bars − 1 — нижняя граница, если ряд начинается в 00:00
    (иначе первая неполная сессия тоже без входов).
    """
    if anchor == "rolling":
        return 2 * int(w) - 2
    return int(min_session_bars) - 1


def vwap(df: pd.DataFrame, *, anchor: str, w: int | None = None) -> pd.DataFrame:
    """vwap, dev, session_bar (номер рабочего бара в сессии; для rolling — NaN)
    по рабочим барам; на барах простоя — NaN."""
    down = downtime_mask(df)
    work = df.loc[~down, ["close", "volume", "turnover"]].astype("float64")
    if anchor == "session":
        key = work.index.floor("1D")
        g = work.groupby(key)
        cv = g["volume"].cumsum()
        ct = g["turnover"].cumsum()
        vw = ct / cv.where(cv > 0)
        dev = (work["close"] - vw).groupby(key).transform(
            lambda s: s.expanding(min_periods=2).std(ddof=1))
        sb = g.cumcount() + 1
        out = pd.DataFrame({"vwap": vw, "dev": dev, "session_bar": sb.astype("float64")})
    elif anchor == "rolling":
        cv = work["volume"].rolling(w, min_periods=w).sum()
        ct = work["turnover"].rolling(w, min_periods=w).sum()
        vw = ct / cv.where(cv > 0)
        dev = (work["close"] - vw).rolling(w, min_periods=w).std(ddof=1)
        out = pd.DataFrame({"vwap": vw, "dev": dev, "session_bar": np.nan}, index=work.index)
    else:
        raise ValueError(f"anchor ∈ {ANCHORS}")
    return out.reindex(df.index)


def _entry_gate(df: pd.DataFrame, v: pd.DataFrame, anchor: str,
                min_session_bars: int | None) -> np.ndarray:
    """Где разрешены входы: rolling — везде; session — с min_session_bars-го
    рабочего бара сессии и не в первой неполной сессии ряда."""
    if anchor == "rolling":
        return np.ones(len(df), bool)
    gate = (v["session_bar"] >= min_session_bars).to_numpy().copy()
    down = downtime_mask(df)
    work_idx = df.index[~down]
    if len(work_idx) and work_idx[0] != work_idx[0].floor("1D"):
        first_day = work_idx[0].floor("1D")
        gate = gate & np.asarray(df.index.floor("1D") != first_day)
    return gate


def target_move(df: pd.DataFrame, *, anchor: str, k: float, e: float,
                w: int | None = None, min_session_bars: int | None = None) -> pd.Series:
    """(k − e) · dev / close — для cost_to_target."""
    _validate(anchor, k, e, w, min_session_bars)
    v = vwap(df, anchor=anchor, w=w)
    return (k - e) * v["dev"] / df["close"].astype("float64")


def signal(df: pd.DataFrame, *, anchor: str, k: float, e: float,
           w: int | None = None, min_session_bars: int | None = None) -> pd.Series:
    """Позиция ∈ {−1, 0, 1} на индексе df; на прогреве 0."""
    _validate(anchor, k, e, w, min_session_bars)
    v = vwap(df, anchor=anchor, w=w)
    c = df["close"].to_numpy(dtype="float64")
    vw, dev = v["vwap"].to_numpy(), v["dev"].to_numpy()
    ok = ~np.isnan(vw) & ~np.isnan(dev) & (np.nan_to_num(dev) > 0)
    vw0, dev0 = np.where(ok, vw, 0.0), np.where(ok, dev, 0.0)
    below = ok & (c < vw0 - k * dev0)
    above = ok & (c > vw0 + k * dev0)
    near = ok & (np.abs(c - vw0) <= e * dev0)
    gate = _entry_gate(df, v, anchor, min_session_bars)
    # встречный сигнал закрывает всегда; новый вход — только где разрешён
    pos = run_state_machine(below & gate, above & gate, near | above, near | below,
                            frozen=downtime_mask(df))
    return pd.Series(pos, index=df.index, name="pos")


def carry_kwargs(tf: str, cfg: dict, params: dict) -> dict | None:
    """Для описательной статистики переноса через 00:00: только якорь session."""
    if params.get("anchor") != "session":
        return None
    return {"early_bars": int(params["min_session_bars"])}


def session_carry_stats(df: pd.DataFrame, pos: pd.Series, *,
                        min_session_bars: int) -> dict[str, float]:
    """Перенос сделок через 00:00 UTC (описательно): metrics.session_carry_stats
    с early_bars = min_session_bars."""
    from src.backtest.metrics import session_carry_stats as _scs
    return _scs(df.index, pos, early_bars=min_session_bars, is_downtime=downtime_mask(df))
