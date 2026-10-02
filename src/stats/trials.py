"""Журнал попыток: results/trials_log.csv (в git), только дописывание.

Типы записей (docs/preregistration.md §8):
- train — каждая ранжированная конфигурация в фолде (N_pass; исключённые
  по издержкам не пишутся), для аудита уровня 1;
- attempt — попытка уровня 2 (вариант дизайна × стратегия, 1h, slippage 0),
  только они входят в N для DSR вердиктов;
- sensitivity — прогоны slippage 0.0005 / 0.001 (не попытки);
- diagnostic — диагностика без отбора (например, валовая и чистая
  доходность всей сетки, src.cv.diagnostics); в N не входит.
Прогоны 15m пишутся как train с вариантом tf=15m; attempt для них нет.
Каждая запись — с меткой времени и коротким хэшем git.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

COLUMNS = ["time_utc", "git", "kind", "variant", "strategy", "tf", "symbol", "fold",
           "params", "sharpe", "n_obs", "note"]
KINDS = ("train", "attempt", "sensitivity", "diagnostic")


def git_hash() -> str:
    """Короткий хэш HEAD; 'nogit' вне репозитория."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "nogit"


def log_rows(path: str | Path, rows: pd.DataFrame) -> int:
    """Дописать строки (колонки COLUMNS без time_utc/git — добавятся). Возвращает число строк."""
    if rows.empty:
        return 0
    bad = set(rows["kind"]) - set(KINDS)
    if bad:
        raise ValueError(f"неизвестный тип записи: {bad}")
    out = rows.copy()
    out.insert(0, "git", git_hash())
    out.insert(0, "time_utc", pd.Timestamp.now(tz="UTC").isoformat())
    out["params"] = out["params"].map(lambda p: p if isinstance(p, str) else json.dumps(p))
    out = out[COLUMNS]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(out)


def log_trial(path: str, *, kind: str, variant: str, strategy: str, tf: str, symbol: str,
              params: dict, sharpe: float, n_obs: int, fold: str = "", note: str = "") -> None:
    """Дописать одну запись."""
    log_rows(path, pd.DataFrame([{"kind": kind, "variant": variant, "strategy": strategy,
                                  "tf": tf, "symbol": symbol, "fold": fold, "params": params,
                                  "sharpe": sharpe, "n_obs": n_obs, "note": note}]))


def read_trials(path: str | Path) -> pd.DataFrame:
    """Прочитать журнал (пустой DataFrame, если файла нет)."""
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)


def n_trials(path: str | Path, **filters) -> int:
    """Число записей с фильтрами по колонкам (например kind='attempt', tf='1h')."""
    t = read_trials(path)
    for k, v in filters.items():
        t = t[t[k] == v]
    return len(t)


def train_rows(wf, gd, variant: str, kind: str = "train") -> pd.DataFrame:
    """Записи по результату walk_forward: ранжированные (N_pass) конфигурации
    каждого фолда; n_obs — длина train-окна в барах."""
    from src.cv.grid import window_rows

    rows = []
    for s in wf.selections:
        r = window_rows(gd.index, s.train_start, s.fold.train_end)
        for j in [int(i) for i in s.passed.nonzero()[0]]:
            rows.append({"kind": kind, "variant": variant, "strategy": gd.strategy,
                         "tf": gd.tf, "symbol": gd.symbol, "fold": str(s.fold.val_start.date()),
                         "params": gd.grid_params[j], "sharpe": float(s.train_sharpe[j]),
                         "n_obs": r.stop - r.start, "note": ""})
    return pd.DataFrame(rows, columns=[c for c in COLUMNS if c not in ("time_utc", "git")])
