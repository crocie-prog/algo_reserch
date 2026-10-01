"""Шаг 2 этапа 1: хранение parquet."""
import os

import pandas as pd
import pytest

from src.data import store
from tests.fakes import make_bars


def test_roundtrip_single_file(tmp_path):
    df = make_bars("2023-01-01", 100, "1h")
    assert store.write(tmp_path, "X", "1h", df) == 100
    assert (tmp_path / "1h" / "X.parquet").exists()
    pd.testing.assert_frame_equal(store.read(tmp_path, "X", "1h"), df, check_freq=False)


def test_1m_partitioned_by_year(tmp_path):
    df = make_bars("2022-12-31 23:00", 180, "1min")
    store.write(tmp_path, "X", "1m", df)
    files = store.partitions(tmp_path, "X", "1m")
    assert [f.stem for f in files] == ["2022", "2023"]
    pd.testing.assert_frame_equal(store.read(tmp_path, "X", "1m"), df, check_freq=False)
    assert store.last_timestamp(tmp_path, "X", "1m") == df.index[-1]


def test_1m_incremental_rewrites_only_touched_year(tmp_path):
    df = make_bars("2022-12-31 23:00", 180, "1min")
    store.write(tmp_path, "X", "1m", df.iloc[:120])
    p2022 = tmp_path / "1m" / "X" / "2022.parquet"
    mtime = p2022.stat().st_mtime_ns
    store.write(tmp_path, "X", "1m", df.iloc[120:])
    assert p2022.stat().st_mtime_ns == mtime
    pd.testing.assert_frame_equal(store.read(tmp_path, "X", "1m"), df, check_freq=False)


def test_read_filters_range(tmp_path):
    df = make_bars("2022-12-31 23:00", 180, "1min")
    store.write(tmp_path, "X", "1m", df)
    got = store.read(tmp_path, "X", "1m", df.index[10], df.index[20])
    assert len(got) == 11


def test_identical_overlap_is_merged(tmp_path):
    df = make_bars("2023-01-01", 100, "1h")
    store.write(tmp_path, "X", "1h", df.iloc[:60])
    assert store.write(tmp_path, "X", "1h", df.iloc[50:]) == 40
    assert len(store.read(tmp_path, "X", "1h")) == 100


def test_conflicting_overlap_raises(tmp_path):
    df = make_bars("2023-01-01", 100, "1h")
    store.write(tmp_path, "X", "1h", df.iloc[:60])
    changed = df.iloc[55:].copy()
    changed.iloc[0, 3] += 1.0
    with pytest.raises(store.RevisionError):
        store.write(tmp_path, "X", "1h", changed)
    assert len(store.read(tmp_path, "X", "1h")) == 60      # файл не тронут


def test_overwrite_from_replaces(tmp_path):
    df = make_bars("2023-01-01", 100, "1h")
    store.write(tmp_path, "X", "1h", df.iloc[:60])
    changed = df.iloc[55:].copy()
    changed.iloc[0, 3] += 1.0
    store.write(tmp_path, "X", "1h", changed, overwrite_from=df.index[55])
    got = store.read(tmp_path, "X", "1h")
    assert got.iloc[55, 3] == changed.iloc[0, 3] and len(got) == 100


def test_atomic_write_keeps_old_on_failure(tmp_path, monkeypatch):
    df = make_bars("2023-01-01", 10, "1h")
    store.write(tmp_path, "X", "1h", df.iloc[:5])

    def boom(src, dst):
        raise OSError("сбой между tmp и replace")

    monkeypatch.setattr(store.os, "replace", boom)
    with pytest.raises(OSError):
        store.write(tmp_path, "X", "1h", df.iloc[5:])
    monkeypatch.undo()
    assert len(store.read(tmp_path, "X", "1h")) == 5
    assert not [f for f in os.listdir(tmp_path / "1h") if ".tmp-" in f]


def test_funding_path(tmp_path):
    s = pd.DataFrame({"funding_rate": [1e-4, -2e-4]},
                     index=pd.date_range("2023-01-01", periods=2, freq="8h", tz="UTC"))
    store.write(tmp_path, "X", "funding", s)
    assert (tmp_path / "funding" / "X.parquet").exists()
