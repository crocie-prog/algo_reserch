"""Purged K-fold с embargo (López de Prado, AFML, гл. 7). Этап 6.

Нужен при перекрывающихся позициях: из обучения удаляются наблюдения,
чей горизонт пересекается с валидацией, плюс embargo после неё.
"""
from __future__ import annotations

import pandas as pd


def purged_kfold(index: pd.DatetimeIndex, t1: pd.Series, *, n_splits: int,
                 embargo: float) -> list[tuple[pd.Index, pd.Index]]:
    """Разбиения (train_idx, val_idx).

    Args:
        t1: время окончания горизонта каждого наблюдения.
        embargo: доля выборки, исключаемая после валидационного блока.
    """
    raise NotImplementedError
