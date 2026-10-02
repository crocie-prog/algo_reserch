"""Проверки 1m по годовым партициям совпадают с проверками целого ряда."""
import pandas as pd
import pytest

from src.data import quality, store
from tests.fakes import make_bars


@pytest.fixture
def m1(cfg):
    """1m около трёх стыков лет с проблемами на стыках."""
    cfg["quality"]["sigma_window"] = 50
    parts = [make_bars("2021-12-31 20:00", 480, "1min", seed=1),
             make_bars("2022-12-31 22:00", 480, "1min", seed=2),
             make_bars("2023-12-31 23:00", 180, "1min", seed=3)]
    df = pd.concat(parts)
    df = df[~df.index.duplicated()]
    # нулевой объём через стык 2022/2023: 3 последних бара 2022 + 2 первых 2023
    t0 = pd.Timestamp("2022-12-31 23:57", tz="UTC")
    df.loc[t0:t0 + pd.Timedelta(minutes=4), "volume"] = 0.0
    # high < low через тот же стык (2 + 1 бар)
    t1 = pd.Timestamp("2022-12-31 23:58", tz="UTC")
    sl = slice(t1, t1 + pd.Timedelta(minutes=2))
    df.loc[sl, "high"] = df.loc[sl, "low"] * 0.99
    # выброс на 3-м баре 2023 (σ считается по хвосту 2022)
    t2 = pd.Timestamp("2023-01-01 00:03", tz="UTC")
    df.loc[t2:, ["open", "high", "low", "close"]] *= 1.5
    # пропуск через стык 2023/2024: 23:55 … 00:04
    df = df.drop(df.loc["2023-12-31 23:55":"2024-01-01 00:04"].index)
    store.write(cfg["paths"]["clean"], "X", "1m", df)
    assert [f.stem for f in store.partitions(cfg["paths"]["clean"], "X", "1m")] == \
        ["2021", "2022", "2023", "2024"]
    return df


def test_bars_stream_equals_whole(cfg, m1):
    iss_w, s_w = quality.check_bars(m1, "X", "1m", cfg)
    iss_s, s_s = quality.check_bars_stream(
        quality.iter_partitions(cfg["paths"]["clean"], "X", "1m"), "X", "1m", cfg)
    pd.testing.assert_frame_equal(iss_s.reset_index(drop=True), iss_w.reset_index(drop=True),
                                  check_dtype=False)
    assert s_s == s_w
    # сценарии на стыках действительно попали в отчёт
    zv = iss_s[iss_s.check == "zero_volume"]
    assert len(zv) == 1 and zv.iloc[0]["n_bars"] == 5
    assert pd.Timestamp("2023-01-01 00:03", tz="UTC") in set(iss_s.loc[iss_s.check == "outlier", "start"])
    gap = iss_s[(iss_s.check == "gap") & (iss_s.start == pd.Timestamp("2023-12-31 23:55", tz="UTC"))]
    assert gap.iloc[0]["n_bars"] == 10


def test_consistency_stream_equals_whole(cfg, m1):
    h1 = quality.aggregate(m1, "1h").drop(columns="n")
    h1.iloc[2, 1] *= 1.001                                   # расхождение high
    h1.loc[pd.Timestamp("2023-06-01", tz="UTC")] = h1.iloc[0].values   # бар без 1m
    h1 = h1.sort_index()
    iss_w, s_w = quality.check_consistency(m1, h1, "1m", "1h", "X", cfg)
    iss_s, s_s = quality.check_consistency_stream(
        quality.iter_partitions(cfg["paths"]["clean"], "X", "1m"), h1, "1m", "1h", "X", cfg)
    pd.testing.assert_frame_equal(iss_s.reset_index(drop=True), iss_w.reset_index(drop=True),
                                  check_dtype=False)
    assert s_s == s_w
    assert s_s["agg_mismatch"] == 1 and s_s["agg_incomplete"] >= 1
    assert s_s["agg_coarse_without_fine"] == 1


def test_run_all_per_symbol_keeps_other_rows(cfg):
    for sym, seed in (("A", 1), ("B", 2)):
        m = make_bars("2021-10-15", 1440, "1min", seed=seed)
        store.write(cfg["paths"]["raw"], sym, "1m", m)
        store.write(cfg["paths"]["raw"], sym, "1h", quality.aggregate(m, "1h").drop(columns="n"))
    quality.run_all(cfg, symbols=["A"])
    quality.run_all(cfg, symbols=["B"])
    quality.run_all(cfg, symbols=["A"])                      # повтор не дублирует A
    summ = pd.read_csv(cfg["paths"]["data_quality_summary"])
    assert sorted(summ["symbol"].unique()) == ["A", "B"]
    assert len(summ[(summ.symbol == "A") & (summ.kind == "bars") & (summ.tf == "1m")]) == 1
