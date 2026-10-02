"""Санити-тесты движка (CLAUDE.md, этап 2):

1. buy&hold совпадает с ценой;
2. случайный сигнал ≈ −издержки;
3. сигнал, сдвинутый на −1 бар, даёт подозрительно хороший результат.

Синтетика — детерминированная (случайное блуждание без дрейфа). Варианты
на реальных данных (BTCUSDT 1h, train) — метка data, пропускаются без clean.
Функции stat_* возвращают числа и используются тестами и сводкой.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import engine
from src.backtest.metrics import metrics
from src.config import load_config

P_1H = 8760
FEE = 0.001


def random_walk(n=200_000, sigma=0.01, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    return pd.DataFrame({"close": close}, index=idx)


def random_positions(index, p_switch=0.05, seed=7):
    """Марковские случайные позиции {−1, 0, 1}, независимые от цен."""
    rng = np.random.default_rng(seed)
    n = len(index)
    draws = rng.choice([-1.0, 0.0, 1.0], n)
    switch = rng.random(n) < p_switch
    switch[0] = True
    pos = pd.Series(np.where(switch, draws, np.nan), index=index).ffill()
    return pos


def real_btc_1h():
    try:
        from src.data.load import load
        df = load("BTCUSDT", "1h")                      # train по умолчанию
        f = __import__("src.data.load", fromlist=["load_funding"]).load_funding("BTCUSDT")
        return df, f
    except Exception:
        return None, None


def stat_buy_hold(df, fee):
    pos = pd.Series(1.0, index=df.index)
    bt = engine.run(df, pos, fee_per_side=fee)
    expected = (1 - fee) * df["close"].iloc[-1] / df["close"].iloc[0]
    return {"equity_end": bt["equity"].iloc[-1], "price_ratio_x_(1-fee)": expected,
            "rel_err": bt["equity"].iloc[-1] / expected - 1}


def stat_random(df, funding=None, tf="1h", fee=FEE):
    pos = random_positions(df.index)
    bt = engine.run(df, pos, fee_per_side=fee, funding=funding, tf=tf)
    n = len(bt)
    g = bt["gross"].to_numpy()
    cost = (bt["fee"] + bt["slippage"] + bt["funding"]).to_numpy()
    se = g.std(ddof=1) / np.sqrt(n)
    m = metrics(bt, periods_per_year=P_1H)
    return {"n": n, "mean_gross": g.mean(), "se_gross": se, "t_gross": g.mean() / se,
            "mean_net": bt["net"].mean(), "mean_cost": cost.mean(),
            "net_plus_cost_over_se": (bt["net"].mean() + cost.mean()) / se,
            "sharpe_net": m["sharpe"], "turnover_per_year": m["turnover_per_year"]}


def stat_shift(df, fee):
    ret = df["close"].pct_change()
    lead = np.sign(ret.shift(-1)).fillna(0.0)          # знает доходность следующего бара
    lag = np.sign(ret).fillna(0.0)                      # честный: знает только прошлое
    s_lead = metrics(engine.run(df, lead, fee_per_side=fee), periods_per_year=P_1H)
    s_lag = metrics(engine.run(df, lag, fee_per_side=fee), periods_per_year=P_1H)
    return {"sharpe_lead": s_lead["sharpe"], "ann_return_lead": s_lead["ann_return"],
            "sharpe_lag": s_lag["sharpe"], "se_sharpe": np.sqrt(P_1H / len(df))}


# ── 1. buy&hold ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("fee", [0.0, FEE])
def test_buy_hold_matches_price_synthetic(fee):
    r = stat_buy_hold(random_walk(20_000), fee)
    assert abs(r["rel_err"]) < 1e-10


@pytest.mark.data
def test_buy_hold_matches_price_real():
    df, _ = real_btc_1h()
    if df is None:
        pytest.skip("нет clean BTCUSDT 1h")
    r = stat_buy_hold(df, 0.0)
    assert abs(r["rel_err"]) < 1e-10


# ── 2. случайный сигнал ≈ −издержки ──────────────────────────────────────

def test_random_signal_costs_synthetic():
    r = stat_random(random_walk())
    assert abs(r["t_gross"]) < 4                       # валовой результат ≈ 0
    assert abs(r["net_plus_cost_over_se"]) < 4         # net ≈ −издержки
    assert r["mean_net"] < 0


@pytest.mark.data
def test_random_signal_costs_real():
    df, f = real_btc_1h()
    if df is None:
        pytest.skip("нет clean BTCUSDT 1h")
    r = stat_random(df, funding=f)
    assert abs(r["t_gross"]) < 4
    assert abs(r["net_plus_cost_over_se"]) < 4
    assert r["mean_net"] < 0


# ── 3. сдвиг на −1 бар ловится ───────────────────────────────────────────

@pytest.mark.parametrize("fee", [0.0, FEE])
def test_shift_minus_one_is_suspiciously_good(fee):
    r = stat_shift(random_walk(), fee)
    assert r["sharpe_lead"] > 20                        # абсурдно высоко → утечка
    if fee == 0.0:
        assert abs(r["sharpe_lag"]) < 4 * r["se_sharpe"]   # честная версия ≈ 0
