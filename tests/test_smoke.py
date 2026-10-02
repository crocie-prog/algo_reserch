"""Этап 0: проект собирается — модули импортируются, config.yaml читается."""
import importlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    "src.config",
    "src.data.bybit", "src.data.store", "src.data.fetch", "src.data.quality",
    "src.data.universe", "src.data.load",
    "src.backtest.engine", "src.backtest.costs", "src.backtest.metrics",
    "src.strategies._common",
    "src.strategies.s1_zscore", "src.strategies.s2_bollinger",
    "src.strategies.s3_donchian", "src.strategies.s4_supertrend",
    "src.strategies.s5_pivot", "src.strategies.s6_vwap",
    "src.cv.walkforward", "src.cv.purged", "src.cv.cpcv",
    "src.portfolio.weights", "src.portfolio.aggregate",
    "src.stats.sharpe", "src.stats.lw", "src.stats.bootstrap",
    "src.stats.dsr", "src.stats.ce", "src.stats.trials", "src.stats.neff",
    "src.cv.grid", "src.cv.permute", "src.cv.stage5", "src.cv.report", "src.cv.diagnostics",
    "src.cv.stage6", "src.data.select_universe", "src.data.liquidity",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_config_basic():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["universe"]["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert cfg["periods"]["train_end"] < cfg["periods"]["test_start"]
    assert cfg["costs"]["fee_per_side"] == 0.001
    assert cfg["costs"]["rf"] == 0.0
    assert set(cfg["selection"]["min_trades_per_year"]) == {"1h", "15m", "1d"}


@pytest.mark.parametrize("name", [m for m in MODULES if m.startswith("src.strategies.s")])
def test_strategy_interface(name):
    mod = importlib.import_module(name)
    assert callable(mod.signal) and callable(mod.warmup)
    assert isinstance(mod.WARMUP_PARAMS, tuple)


def test_grids_match_preregistration():
    import math
    from src.config import load_config
    g = load_config()["grids"]
    sizes = {s: math.prod(len(v) for v in g[s]["1h"].values()) if g[s]["1h"] else 1 for s in g}
    assert sizes == {"s1_zscore": 72, "s3_donchian": 10, "s4_supertrend": 20,
                     "s5_pivot": 1, "s6_vwap": 16}
    assert sum(sizes.values()) == 119
    assert "s2_bollinger" not in g
    # 15m: те же горизонты в часах (окна ×4)
    assert g["s1_zscore"]["15m"]["w"] == [4 * w for w in g["s1_zscore"]["1h"]["w"]]
    assert g["s3_donchian"]["15m"]["w"] == [4 * w for w in g["s3_donchian"]["1h"]["w"]]
    assert g["s4_supertrend"]["15m"]["n"] == [4 * n for n in g["s4_supertrend"]["1h"]["n"]]
