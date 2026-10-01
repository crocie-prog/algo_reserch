"""Чтение config.yaml и общие утилиты таймфреймов.

Все изменяемые числа (периоды, комиссии, окна, сетки) берутся отсюда,
в коде констант нет. Относительные пути из раздела paths разрешаются
от папки, где лежит config.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"

_TF_MINUTES = {"1m": 1, "15m": 15, "1h": 60, "1d": 1440}


def tf_minutes(tf: str) -> int:
    """Длительность ТФ в минутах."""
    try:
        return _TF_MINUTES[tf]
    except KeyError:
        raise ValueError(f"неизвестный таймфрейм: {tf}") from None


def tf_delta(tf: str) -> pd.Timedelta:
    """Длительность ТФ как Timedelta."""
    return pd.Timedelta(minutes=tf_minutes(tf))


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Прочитать config.yaml и проверить обязательные разделы.

    Проверки: train_end < test_start; символы и таймфреймы не пусты;
    fee_per_side ≥ 0. Пути из paths заменяются абсолютными (ключ _root —
    папка конфига).

    Raises:
        ValueError: при нарушении проверок.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent
    cfg["_root"] = root
    cfg["paths"] = {k: (root / v) for k, v in cfg["paths"].items()}

    p = cfg["periods"]
    if pd.Timestamp(p["train_end"]) >= pd.Timestamp(p["test_start"]):
        raise ValueError("periods.train_end должен быть раньше test_start")
    if not cfg["universe"]["symbols"]:
        raise ValueError("universe.symbols пуст")
    for tf in cfg["timeframes"]["download"]:
        tf_minutes(tf)
    if cfg["costs"]["fee_per_side"] < 0:
        raise ValueError("costs.fee_per_side < 0")
    return cfg


def periods_per_year(tf: str, cfg: dict[str, Any]) -> float:
    """Число баров в году для аннуализации: days_per_year · 1440 / минуты ТФ.

    Пример: 1h → 365 · 24 = 8760.
    """
    return cfg["annualization"]["days_per_year"] * 1440 / tf_minutes(tf)
