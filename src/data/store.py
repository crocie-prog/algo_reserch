"""Хранение parquet.

Раскладка (одинакова для raw и clean):
- 15m/1h/1d: <root>/<tf>/<SYMBOL>.parquet — один файл на пару;
- 1m: <root>/1m/<SYMBOL>/<YYYY>.parquet — партиции по годам; при докачке
  перезаписывается только партиция года, в который попадают новые бары;
- funding: <root>/funding/<SYMBOL>.parquet.

Запись атомарная: временный файл в той же папке + os.replace.
Слияние со старыми данными: совпадающие строки сливаются, различающиеся —
RevisionError (пересмотр биржей), если явно не задан overwrite_from.
Файлы в data/raw/ вручную не изменяются.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PARTITIONED_TFS = {"1m"}


class RevisionError(RuntimeError):
    """Новые данные расходятся с уже сохранёнными на тех же метках."""


def _is_partitioned(tf: str) -> bool:
    return tf in PARTITIONED_TFS


def raw_path(root: str | Path, symbol: str, tf: str, year: int | None = None) -> Path:
    """Путь к файлу: tf='funding' — история funding; для 1m нужен year."""
    root = Path(root)
    if _is_partitioned(tf):
        if year is None:
            raise ValueError("для 1m нужен год партиции")
        return root / tf / symbol / f"{year}.parquet"
    return root / tf / f"{symbol}.parquet"


def partitions(root: str | Path, symbol: str, tf: str) -> list[Path]:
    """Существующие файлы ряда по возрастанию (для 1m — по годам)."""
    if _is_partitioned(tf):
        d = Path(root) / tf / symbol
        return sorted(d.glob("[0-9][0-9][0-9][0-9].parquet")) if d.exists() else []
    p = raw_path(root, symbol, tf)
    return [p] if p.exists() else []


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).as_unit("ns")
    return df


def read_file(path: str | Path) -> pd.DataFrame | None:
    """Прочитать один parquet; None, если файла нет."""
    path = Path(path)
    if not path.exists():
        return None
    return _normalize(pd.read_parquet(path))


def read(root: str | Path, symbol: str, tf: str, start: pd.Timestamp | None = None,
         end: pd.Timestamp | None = None) -> pd.DataFrame | None:
    """Склеить ряд (все партиции) и отфильтровать [start, end] по времени открытия.

    Для 1m читаются только партиции лет, пересекающихся с [start, end].
    """
    files = partitions(root, symbol, tf)
    if _is_partitioned(tf):
        y0 = start.year if start is not None else -1
        y1 = end.year if end is not None else 10**6
        files = [f for f in files if y0 <= int(f.stem) <= y1]
    if not files:
        return None
    df = pd.concat([read_file(f) for f in files])
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]
    return df


def last_timestamp(root: str | Path, symbol: str, tf: str) -> pd.Timestamp | None:
    """Последняя сохранённая метка — точка инкрементальной докачки."""
    files = partitions(root, symbol, tf)
    if not files:
        return None
    df = read_file(files[-1])
    return df.index[-1] if len(df) else None


def write_atomic(df: pd.DataFrame, path: str | Path) -> None:
    """Записать parquet атомарно: tmp в той же папке → os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        df.to_parquet(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def merge(old: pd.DataFrame | None, new: pd.DataFrame,
          overwrite_from: pd.Timestamp | None = None) -> pd.DataFrame:
    """Объединить старые и новые строки.

    Перекрывающиеся метки: при равных значениях — одна строка; при разных —
    RevisionError, кроме меток ≥ overwrite_from, где побеждают новые данные.
    """
    new = _normalize(new)
    if old is None or old.empty:
        out = new
    else:
        common = old.index.intersection(new.index)
        if overwrite_from is not None:
            common = common[common < overwrite_from]
        if len(common):
            a = old.loc[common].to_numpy(dtype="float64")
            b = new.loc[common, old.columns].to_numpy(dtype="float64")
            bad = ~np.isclose(a, b, rtol=0, atol=0, equal_nan=True).all(axis=1)
            if bad.any():
                raise RevisionError(
                    f"{bad.sum()} строк расходятся с сохранёнными, первая: {common[bad][0]}")
        keep_old = old if overwrite_from is None else old[old.index < overwrite_from]
        keep_old = keep_old[~keep_old.index.isin(new.index)]
        out = pd.concat([keep_old, new])
    out = out.sort_index()
    if out.index.has_duplicates:
        raise RevisionError("дубли меток после слияния")
    return out


def write(root: str | Path, symbol: str, tf: str, new: pd.DataFrame,
          overwrite_from: pd.Timestamp | None = None) -> int:
    """Слить новые строки с сохранёнными и записать атомарно.

    Для 1m — по партициям лет, затронутых новыми данными.

    Returns:
        Число добавленных меток (без учёта перезаписанных).
    """
    if new.empty:
        return 0
    new = _normalize(new)
    added = 0
    if _is_partitioned(tf):
        groups = new.groupby(new.index.year)
        targets = [(raw_path(root, symbol, tf, int(y)), g) for y, g in groups]
    else:
        targets = [(raw_path(root, symbol, tf), new)]
    for path, part in targets:
        old = read_file(path)
        out = merge(old, part, overwrite_from)
        added += len(out) - (0 if old is None else len(old))
        write_atomic(out, path)
    return added
