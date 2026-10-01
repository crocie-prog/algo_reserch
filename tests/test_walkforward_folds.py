"""Фолды walk-forward: квартальный шаг, минимум 12 месяцев, test не трогается.

Спецификация до реализации; снять skip на этапе 5–6.
"""
import pandas as pd
import pytest

from src.cv.walkforward import folds

pytestmark = pytest.mark.skip(reason="реализация folds() — этап 5")

IDX = pd.date_range("2021-01-01", "2025-12-31 23:00", freq="1h", tz="UTC")
TRAIN_END = pd.Timestamp("2024-01-01", tz="UTC")   # граница: train до 2023-12-31 включительно


@pytest.mark.parametrize("scheme", ["expanding", "rolling"])
def test_no_window_after_train_end(scheme):
    fl = folds(IDX, scheme=scheme, step_months=3, min_train_months=12,
               train_end=TRAIN_END, rolling_train_months=12)
    assert fl
    assert all(f.val_end <= TRAIN_END for f in fl)
    assert all(f.train_end <= f.val_start for f in fl)


def test_allow_test_extends():
    fl = folds(IDX, scheme="expanding", step_months=3, min_train_months=12,
               train_end=TRAIN_END, allow_test=True)
    assert any(f.val_end > TRAIN_END for f in fl)


def test_quarter_step_and_min_train():
    fl = folds(IDX, scheme="expanding", step_months=3, min_train_months=12,
               train_end=TRAIN_END)
    assert fl[0].val_start == pd.Timestamp("2022-01-01", tz="UTC")
    starts = [f.val_start for f in fl]
    assert all((b.year - a.year) * 12 + b.month - a.month == 3 for a, b in zip(starts, starts[1:]))


def test_rolling_window_length():
    fl = folds(IDX, scheme="rolling", step_months=3, min_train_months=12,
               train_end=TRAIN_END, rolling_train_months=12)
    for f in fl:
        assert f.train_start == f.val_start - pd.DateOffset(months=12)
