"""Чтение config.yaml.

Все изменяемые числа (периоды, комиссии, окна, сетки) берутся отсюда,
в коде констант нет.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Прочитать config.yaml и проверить обязательные разделы.

    Проверки: train_end < test_start; символы и таймфреймы не пусты;
    fee_per_side ≥ 0.

    Raises:
        ValueError: при нарушении проверок.
    """
    raise NotImplementedError


def periods_per_year(tf: str, cfg: dict[str, Any]) -> float:
    """Число баров в году для аннуализации: days_per_year · 1440 / минуты ТФ.

    Пример: 1h → 365 · 24 = 8760.
    """
    raise NotImplementedError
