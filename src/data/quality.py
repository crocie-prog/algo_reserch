"""Проверки качества данных (по слою clean).

Отчёты:
- paths.data_quality — все найденные проблемы, по строке на эпизод:
  symbol, tf, check, start, end, n_bars, detail;
- paths.data_quality_summary — сводка: kind = bars | consistency | funding.

Пропуски не заполняются — только логируются.

Старшие ТФ в clean — агрегат 1m; родные бары биржи сверяются с ним как
диагностика (строки tf = "1m->1h родной" и т.п.).

Экономия памяти: 1m обрабатывается по годовым партициям (check_bars_stream,
check_consistency_stream), в памяти не больше одной партиции 1m и хвоста
предыдущей. check_bars / check_consistency на целом ряде — эталон для тестов.

CLI: python -m src.data.quality --config config.yaml [--symbols ...]
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import tf_delta
from src.data import clean, store

log = logging.getLogger(__name__)

ISSUE_COLUMNS = ["symbol", "tf", "check", "start", "end", "n_bars", "detail"]


def _runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Отрезки подряд идущих True: (начало, конец, длина)."""
    if not mask.any():
        return []
    rid = (mask != mask.shift()).cumsum()
    g = mask[mask].groupby(rid[mask])
    return [(s.index[0], s.index[-1], len(s)) for _, s in g]


def find_gaps(index: pd.DatetimeIndex, tf: str) -> pd.DataFrame:
    """Пропуски на сетке Δ: start, end (метки отсутствующих баров), n_bars."""
    delta = tf_delta(tf)
    if len(index) < 2:
        return pd.DataFrame(columns=["start", "end", "n_bars"])
    d = index[1:] - index[:-1]
    pos = np.flatnonzero(d > delta)
    return pd.DataFrame({"start": index[pos] + delta, "end": index[pos + 1] - delta,
                         "n_bars": (d[pos] // delta - 1).astype(int)})


def check_bars(df: pd.DataFrame, symbol: str, tf: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Проверки одного ряда.

    - пропуски баров (начало, конец, длительность);
    - дубли индекса, монотонность;
    - нулевой объём (эпизодами);
    - high < max(open, close), low > min(open, close), high < low;
    - выбросы: |log ret| > sigma_mult · σ, σ — скользящее std лог-доходностей
      за sigma_window баров ДО t (без бара t).

    Returns:
        (таблица проблем, сводка).
    """
    q = cfg["quality"]
    rows: list[dict] = []

    def add(check, start, end, n, detail=""):
        rows.append({"symbol": symbol, "tf": tf, "check": check, "start": start,
                     "end": end, "n_bars": n, "detail": detail})

    n_dup = int(df.index.duplicated().sum())
    if n_dup:
        add("duplicates", df.index[0], df.index[-1], n_dup)
    if not df.index.is_monotonic_increasing:
        add("not_monotonic", df.index[0], df.index[-1], 0)

    gaps = find_gaps(df.index, tf)
    for g in gaps.itertuples():
        add("gap", g.start, g.end, g.n_bars)

    for s, e, n in _runs(df["volume"] == 0):
        add("zero_volume", s, e, n)

    bad = ((df["high"] < df[["open", "close"]].max(axis=1))
           | (df["low"] > df[["open", "close"]].min(axis=1))
           | (df["high"] < df["low"]))
    for s, e, n in _runs(bad):
        add("ohlc_inconsistent", s, e, n)

    r = np.log(df["close"]).diff()
    w = int(q["sigma_window"])
    sigma = r.rolling(w, min_periods=w).std().shift(1)
    out = r.abs() > q["sigma_mult"] * sigma
    after_gap = pd.Series(df.index.to_series().diff() > tf_delta(tf), index=df.index)
    for t in df.index[out.fillna(False).to_numpy()]:
        add("outlier", t, t, 1, f"ret={r[t]:.5f}; sigma={sigma[t]:.5f}"
            + ("; после пропуска" if after_gap[t] else ""))

    longest = gaps.loc[gaps["n_bars"].idxmax()] if len(gaps) else None
    summary = {
        "kind": "bars", "symbol": symbol, "tf": tf, "n_bars": len(df),
        "first": df.index[0] if len(df) else None, "last": df.index[-1] if len(df) else None,
        "n_gaps": len(gaps), "missing_bars": int(gaps["n_bars"].sum()) if len(gaps) else 0,
        "longest_gap_bars": int(longest["n_bars"]) if longest is not None else 0,
        "longest_gap_start": longest["start"] if longest is not None else None,
        "zero_volume_bars": int((df["volume"] == 0).sum()),
        "ohlc_bad_bars": int(bad.sum()), "outliers": int(out.sum()), "duplicates": n_dup,
        "downtime_bars": int(df["is_downtime"].sum()) if "is_downtime" in df else None,
    }
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS), summary


def aggregate(fine: pd.DataFrame, coarse_tf: str) -> pd.DataFrame:
    """Агрегат младшего ТФ в корзины старшего (якорь 00:00 UTC).

    O=first, H=max, L=min, C=last, V=sum, turnover=sum, n — число баров в корзине.
    """
    key = fine.index.floor(tf_delta(coarse_tf))
    g = fine.groupby(key)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(),
                        "volume": g["volume"].sum(), "turnover": g["turnover"].sum(),
                        "n": g["close"].size()})
    out.index.name = "open_time"
    return out


def check_consistency(fine: pd.DataFrame, coarse: pd.DataFrame, fine_tf: str,
                      coarse_tf: str, symbol: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Сверка агрегата младшего ТФ с файлом старшего.

    Сравниваются только полные корзины (все бары младшего ТФ на месте):
    O/H/L/C — с допуском price_rtol, volume/turnover — volume_rtol.
    Расхождения — в отчёт, не исправляются. Неполные корзины считаются отдельно.
    """
    q = cfg["quality"]
    ratio = tf_delta(coarse_tf) // tf_delta(fine_tf)
    agg = aggregate(fine, coarse_tf)
    both = agg.index.intersection(coarse.index)
    full = agg.loc[both, "n"] == ratio
    cmp_idx = both[full.to_numpy()]
    a, c = agg.loc[cmp_idx], coarse.loc[cmp_idx]
    bad_cols = pd.DataFrame(index=cmp_idx)
    for col in ["open", "high", "low", "close"]:
        bad_cols[col] = ~np.isclose(a[col], c[col], rtol=q["price_rtol"], atol=0)
    for col in ["volume", "turnover"]:
        bad_cols[col] = ~np.isclose(a[col], c[col], rtol=q["volume_rtol"], atol=1e-12)
    mism = bad_cols.any(axis=1)
    pair = f"{fine_tf}->{coarse_tf}"
    rows = []
    for t in cmp_idx[mism.to_numpy()]:
        cols = [k for k in bad_cols.columns if bad_cols.at[t, k]]
        detail = "; ".join(f"{k}: agg={a.at[t, k]:.10g} file={c.at[t, k]:.10g}" for k in cols)
        rows.append({"symbol": symbol, "tf": pair, "check": "agg_mismatch", "start": t,
                     "end": t, "n_bars": 1, "detail": detail})
    incomplete = both[~full.to_numpy()]     # корзины с пропусками младшего ТФ (они — в gap)
    coarse_only = coarse.index.difference(agg.index)
    summary = {"kind": "consistency", "symbol": symbol, "tf": pair,
               "agg_compared": len(cmp_idx), "agg_mismatch": int(mism.sum()),
               "agg_mismatch_cols": ",".join(k for k in bad_cols.columns if bad_cols[k].any()),
               "agg_incomplete": len(incomplete), "agg_coarse_without_fine": len(coarse_only)}
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS), summary


def iter_partitions(root, symbol: str, tf: str):
    """Партиции ряда по одной (для 1m — по годам; для прочих ТФ — один файл)."""
    for f in store.partitions(root, symbol, tf):
        yield store.read_file(f)


def check_bars_stream(frames, symbol: str, tf: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """То же, что check_bars, но по последовательности партиций.

    В памяти — текущая партиция и хвост предыдущей из sigma_window + 1 баров
    (контекст для σ выбросов и для пропуска на стыке). Эпизоды нулевого
    объёма и несогласованного OHLC, идущие через стык, склеиваются.
    Результат совпадает с check_bars на склеенном ряде.
    """
    q = cfg["quality"]
    w = int(q["sigma_window"])
    delta = tf_delta(tf)
    ctx: pd.DataFrame | None = None
    first = last = None
    n = n_dup = 0
    nonmono = False
    gaps: list[pd.DataFrame] = []
    zero_runs: list[list] = []
    bad_runs: list[list] = []
    outliers: list[dict] = []
    n_zero = n_bad = 0
    n_down: int | None = None

    def merge_runs(acc, new, prev_last, cur_first):
        new = [list(r) for r in new]
        if acc and new and acc[-1][1] == prev_last and new[0][0] == cur_first:
            acc[-1][1], acc[-1][2] = new[0][1], acc[-1][2] + new[0][2]
            new = new[1:]
        acc.extend(new)

    for cur in frames:
        if cur is None or cur.empty:
            continue
        n_dup += int(cur.index.duplicated().sum())
        if not cur.index.is_monotonic_increasing or (ctx is not None and cur.index[0] <= ctx.index[-1]):
            nonmono = True
        full = cur if ctx is None else pd.concat([ctx, cur])
        g = find_gaps(full.index, tf)
        if len(g):
            gaps.append(g[g["end"] + delta >= cur.index[0]])
        prev_last = ctx.index[-1] if ctx is not None else None
        zmask = cur["volume"] == 0
        bmask = ((cur["high"] < cur[["open", "close"]].max(axis=1))
                 | (cur["low"] > cur[["open", "close"]].min(axis=1))
                 | (cur["high"] < cur["low"]))
        n_zero += int(zmask.sum())
        n_bad += int(bmask.sum())
        if "is_downtime" in cur:
            n_down = (n_down or 0) + int(cur["is_downtime"].sum())
        merge_runs(zero_runs, _runs(zmask), prev_last, cur.index[0])
        merge_runs(bad_runs, _runs(bmask), prev_last, cur.index[0])
        r = np.log(full["close"]).diff()
        sigma = r.rolling(w, min_periods=w).std().shift(1)
        out = (r.abs() > q["sigma_mult"] * sigma).fillna(False)
        after_gap = full.index.to_series().diff() > delta
        sel = out[out.index >= cur.index[0]]
        for t in sel.index[sel.to_numpy()]:
            outliers.append({"t": t, "detail": f"ret={r[t]:.5f}; sigma={sigma[t]:.5f}"
                             + ("; после пропуска" if after_gap[t] else "")})
        n += len(cur)
        first = first if first is not None else cur.index[0]
        last = cur.index[-1]
        ctx = cur.iloc[-(w + 1):]

    rows: list[dict] = []

    def add(check, start, end, nb, detail=""):
        rows.append({"symbol": symbol, "tf": tf, "check": check, "start": start,
                     "end": end, "n_bars": nb, "detail": detail})

    if n_dup:
        add("duplicates", first, last, n_dup)
    if nonmono:
        add("not_monotonic", first, last, 0)
    gaps_df = pd.concat(gaps, ignore_index=True) if gaps else \
        pd.DataFrame(columns=["start", "end", "n_bars"])
    for gg in gaps_df.itertuples():
        add("gap", gg.start, gg.end, gg.n_bars)
    for s_, e_, k in zero_runs:
        add("zero_volume", s_, e_, k)
    for s_, e_, k in bad_runs:
        add("ohlc_inconsistent", s_, e_, k)
    for o in outliers:
        add("outlier", o["t"], o["t"], 1, o["detail"])

    longest = gaps_df.loc[gaps_df["n_bars"].astype(int).idxmax()] if len(gaps_df) else None
    summary = {
        "kind": "bars", "symbol": symbol, "tf": tf, "n_bars": n, "first": first, "last": last,
        "n_gaps": len(gaps_df), "missing_bars": int(gaps_df["n_bars"].sum()) if len(gaps_df) else 0,
        "longest_gap_bars": int(longest["n_bars"]) if longest is not None else 0,
        "longest_gap_start": longest["start"] if longest is not None else None,
        "zero_volume_bars": n_zero, "ohlc_bad_bars": n_bad, "outliers": len(outliers),
        "duplicates": n_dup, "downtime_bars": n_down,
    }
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS), summary


def check_consistency_stream(fine_frames, coarse: pd.DataFrame, fine_tf: str, coarse_tf: str,
                             symbol: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """check_consistency по партициям младшего ТФ.

    Стык партиций (1 января 00:00 UTC) совпадает с границей корзин 15m/1h/1d,
    поэтому корзина не может разрезаться между партициями; это проверяется
    явно (RuntimeError, если корзина партиции уже встречалась в предыдущей).
    Результат совпадает с check_consistency на склеенном ряде.
    """
    rows: list[pd.DataFrame] = []
    compared = mism = incomplete = 0
    bad_cols: set[str] = set()
    seen = pd.DatetimeIndex([], tz="UTC").as_unit("ns")
    for part in fine_frames:
        if part is None or part.empty:
            continue
        agg_idx = part.index.floor(tf_delta(coarse_tf)).unique()
        if len(seen) and agg_idx[0] <= seen[-1]:
            raise RuntimeError(f"{symbol} {fine_tf}->{coarse_tf}: корзина {agg_idx[0]} "
                               f"разрезана стыком партиций")
        seen = seen.append(agg_idx)
        sub = coarse[(coarse.index >= agg_idx[0]) & (coarse.index <= agg_idx[-1])]
        iss, s = check_consistency(part, sub, fine_tf, coarse_tf, symbol, cfg)
        rows.append(iss)
        compared += s["agg_compared"]
        mism += s["agg_mismatch"]
        incomplete += s["agg_incomplete"]
        bad_cols |= set(filter(None, s["agg_mismatch_cols"].split(",")))
    order = ["open", "high", "low", "close", "volume", "turnover"]
    summary = {"kind": "consistency", "symbol": symbol, "tf": f"{fine_tf}->{coarse_tf}",
               "agg_compared": compared, "agg_mismatch": mism,
               "agg_mismatch_cols": ",".join(c for c in order if c in bad_cols),
               "agg_incomplete": incomplete,
               "agg_coarse_without_fine": len(coarse.index.difference(seen))}
    rep = pd.concat([r for r in rows if len(r)], ignore_index=True) if any(len(r) for r in rows) \
        else pd.DataFrame(columns=ISSUE_COLUMNS)
    return rep, summary


def _merge_report(path, new: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Заменить в CSV строки обработанных символов, строки прочих сохранить."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, low_memory=False)
        old = old[~old["symbol"].isin(symbols)]
        new = pd.concat([old, new], ignore_index=True) if len(old) else new
    new.to_csv(path, index=False)
    return new


def run_all(cfg: dict, symbols: list[str] | None = None, tfs: list[str] | None = None) -> pd.DataFrame:
    """Построить clean и отчёт по символам, по одному символу за раз.

    clean 1m — по одной годовой партиции; clean 15m/1h/1d и производные ТФ
    (timeframes.derived, напр. 4h) — агрегацией clean 1m (см. src.data.clean). Диагностика: родные бары биржи из raw сверяются
    с агрегатом clean 1m (quality.native_vs_1m), расхождения — в отчёт.
    В отчётах заменяются только строки обработанных символов.

    Returns:
        Сводка по обработанным символам (kind = bars | consistency | funding).
    """
    symbols = symbols or cfg["universe"]["symbols"]
    tfs = tfs or cfg["timeframes"]["download"] + cfg["timeframes"].get("derived", [])
    croot, rroot = cfg["paths"]["clean"], cfg["paths"]["raw"]
    all_iss, all_summ = [], []
    for symbol in symbols:
        issues, summ = [], []
        for tf in cfg["timeframes"]["download"] + cfg["timeframes"].get("derived", []):
            info = clean.build_clean(cfg, symbol, tf)
            if info["n_bars"] == 0 or tf not in tfs:
                continue
            frames = iter_partitions(croot, symbol, tf)
            iss, s = check_bars_stream(frames, symbol, tf, cfg)
            check = "dropped_incomplete_first" if tf == clean.BASE_TF else "dropped_incomplete_bin"
            s["n_dropped"] = info["n_dropped"]
            drop_rows = pd.DataFrame([{"symbol": symbol, "tf": tf, "check": check,
                                       "start": t, "end": t, "n_bars": 1,
                                       "detail": "не покрыт торгами / минутками целиком"}
                                      for t in info["dropped"]], columns=ISSUE_COLUMNS)
            issues += [iss, drop_rows]
            summ.append(s)
            log.info("%s %s: баров %d, пропусков %d, простой %s, нулевой объём %d, выбросов %d, "
                     "отброшено неполных %d", symbol, tf, s["n_bars"], s["n_gaps"],
                     s["downtime_bars"], s["zero_volume_bars"], s["outliers"], info["n_dropped"])
        for tf in cfg["quality"]["native_vs_1m"]:
            native = store.read(rroot, symbol, tf)
            if native is None or not store.partitions(croot, symbol, clean.BASE_TF):
                continue
            iss, s = check_consistency_stream(iter_partitions(croot, symbol, clean.BASE_TF),
                                              native, clean.BASE_TF, tf, symbol, cfg)
            s["tf"] = iss["tf"] = f"1m->{tf} родной"
            issues.append(iss)
            summ.append(s)
            log.info("%s 1m -> родной %s: сверено %d, расхождений %d", symbol, tf,
                     s["agg_compared"], s["agg_mismatch"])
        fs = check_funding_symbol(cfg, symbol)
        if fs is not None:
            issues.append(fs[0])
            summ.append(fs[1])
        all_iss += [i for i in issues if len(i)]
        all_summ += summ
    rep = pd.concat(all_iss, ignore_index=True) if all_iss else pd.DataFrame(columns=ISSUE_COLUMNS)
    summary = pd.DataFrame(all_summ)
    _merge_report(cfg["paths"]["data_quality"], rep, symbols)
    _merge_report(cfg["paths"]["data_quality_summary"], summary, symbols)
    return summary


def check_funding(s: pd.Series, symbol: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Проверки истории funding.

    - интервал между начислениями ≠ funding_interval_minutes — эпизодами
      подряд идущих одинаковых интервалов (смена расписания или пропуск);
    - метка не на целом часе UTC (смену интервала ловит предыдущая проверка);
    - дубли меток.
    """
    exp = pd.Timedelta(minutes=int(cfg["quality"]["funding_interval_minutes"]))
    rows: list[dict] = []
    idx = s.index
    d = pd.Series(idx[1:] - idx[:-1], index=idx[1:])
    nonstd = d != exp
    if nonstd.any():
        rid = (d != d.shift()).cumsum()
        for _, g in d[nonstd].groupby(rid[nonstd]):
            rows.append({"symbol": symbol, "tf": "funding", "check": "funding_interval",
                         "start": g.index[0], "end": g.index[-1], "n_bars": len(g),
                         "detail": f"интервал {int(g.iloc[0] / pd.Timedelta(minutes=1))} мин "
                                   f"вместо {int(exp / pd.Timedelta(minutes=1))}"})
    off = idx[idx != idx.floor("1h")]
    for t in off:
        rows.append({"symbol": symbol, "tf": "funding", "check": "funding_offgrid",
                     "start": t, "end": t, "n_bars": 1, "detail": "метка не на целом часе"})
    n_dup = int(idx.duplicated().sum())
    counts = (d / pd.Timedelta(minutes=1)).astype(int).value_counts().sort_index()
    summary = {"kind": "funding", "symbol": symbol, "tf": "funding", "n_bars": len(s),
               "first": idx[0] if len(s) else None, "last": idx[-1] if len(s) else None,
               "funding_nonstd_intervals": int(nonstd.sum()), "funding_offgrid": len(off),
               "duplicates": n_dup,
               "funding_intervals_min": ";".join(f"{k}:{v}" for k, v in counts.items()),
               "funding_max_abs": float(s.abs().max()) if len(s) else None,
               "funding_share_positive": float((s > 0).mean()) if len(s) else None}
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS), summary


def check_funding_symbol(cfg: dict, symbol: str):
    """Проверки funding символа из raw; None — данных нет."""
    df = store.read(cfg["paths"]["raw"], symbol, "funding")
    if df is None or df.empty:
        return None
    iss, s = check_funding(df["funding_rate"], symbol, cfg)
    log.info("%s funding: записей %d, нестандартных интервалов %d, вне сетки %d",
             symbol, s["n_bars"], s["funding_nonstd_intervals"], s["funding_offgrid"])
    return iss, s


def main(argv: list[str] | None = None) -> int:
    """CLI: построить clean и отчёт по одному символу за раз, затем universe.csv."""
    import argparse
    import sys

    from src.config import load_config
    from src.data import universe

    p = argparse.ArgumentParser(prog="python -m src.data.quality")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--symbols", nargs="+")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for symbol in args.symbols or cfg["universe"]["symbols"]:
        run_all(cfg, symbols=[symbol])
    universe.build_universe(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
