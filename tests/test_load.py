"""Шаг 7 этапа 1: load() и защита тестового периода."""
import pandas as pd
import pytest

from src.data import store
from src.data.load import PeriodAccessError, load, load_funding
from tests.fakes import make_bars


@pytest.fixture
def data(cfg):
    h = make_bars("2023-12-30", 24 * 5, "1h")                    # до 2024-01-03
    store.write(cfg["paths"]["clean"], "X", "1h", h)
    m = make_bars("2023-12-31 23:00", 120, "1min")               # через границу года
    store.write(cfg["paths"]["clean"], "X", "1m", m)
    f = pd.DataFrame({"funding_rate": 1e-4},
                     index=pd.date_range("2023-12-31", periods=6, freq="8h", tz="UTC"))
    store.write(cfg["paths"]["raw"], "X", "funding", f)
    return cfg


def test_default_cut_at_train_end(data):
    df = load("X", "1h", cfg=data)
    assert df.index[-1] == pd.Timestamp("2023-12-31 23:00", tz="UTC")


def test_explicit_test_period_requires_flag(data):
    with pytest.raises(PeriodAccessError):
        load("X", "1h", end="2024-01-02", cfg=data)
    df = load("X", "1h", end="2024-01-02", allow_test=True, cfg=data)
    assert df.index[-1] == pd.Timestamp("2024-01-02 23:00", tz="UTC")


def test_end_with_time_inclusive(data):
    df = load("X", "1h", start="2023-12-31 10:00", end="2023-12-31 12:00", cfg=data)
    assert list(df.index.hour) == [10, 11, 12]


def test_1m_partitions_glued(data):
    df = load("X", "1m", cfg=data)
    assert df.index[0] == pd.Timestamp("2023-12-31 23:00", tz="UTC") and len(df) == 60
    df2 = load("X", "1m", allow_test=True, cfg=data)
    assert len(df2) == 120


def test_funding_boundary_belongs_to_train(data):
    f = load_funding("X", cfg=data)
    assert f.index[-1] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    with pytest.raises(PeriodAccessError):
        load_funding("X", end="2024-01-01 08:00", cfg=data)
