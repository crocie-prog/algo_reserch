"""Фолды walk-forward: квартальный шаг, минимум 12 месяцев, test не трогается.
"""
import pandas as pd
import pytest

from src.cv.walkforward import folds

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


@pytest.mark.parametrize("uf,n_folds,first", [
    ("2021-01-01", 8, "2022-01-01"),       # BTC
    ("2021-03-15", 7, "2022-04-01"),       # ETH
    ("2021-10-15", 4, "2023-01-01"),       # SOL
])
def test_usable_from_fold_counts(uf, n_folds, first):
    fl = folds(IDX, scheme="expanding", step_months=3, min_train_months=12,
               train_end=TRAIN_END, usable_from=pd.Timestamp(uf, tz="UTC"))
    assert len(fl) == n_folds and fl[0].val_start == pd.Timestamp(first, tz="UTC")
    assert fl[-1].val_end == TRAIN_END
    assert all(f.train_start == pd.Timestamp(uf, tz="UTC") for f in fl)


def test_rolling_clipped_at_data_start():
    fl = folds(IDX, scheme="rolling", step_months=3, min_train_months=12,
               train_end=TRAIN_END, rolling_train_months=24,
               usable_from=pd.Timestamp("2021-03-15", tz="UTC"))
    assert fl[0].train_start == pd.Timestamp("2021-03-15", tz="UTC")


def test_validation_errors():
    with pytest.raises(ValueError):
        folds(IDX, scheme="sliding", step_months=3, min_train_months=12, train_end=TRAIN_END)
    with pytest.raises(ValueError):
        folds(IDX, scheme="rolling", step_months=3, min_train_months=12, train_end=TRAIN_END)
    with pytest.raises(ValueError):
        folds(IDX, scheme="expanding", step_months=3, min_train_months=60, train_end=TRAIN_END)
