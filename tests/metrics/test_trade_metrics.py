from backlight.metrics import trade_metrics as module
import math
import pandas as pd
import pytest

import backlight.datasource
from backlight.trades import trades as tr
from backlight.trades.trades import make_trades
from backlight.asset.currency import Currency


@pytest.fixture
def symbol():
    return "USDJPY"


@pytest.fixture
def currency_unit():
    return Currency.JPY


@pytest.fixture
def market(symbol, currency_unit):
    data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [9.0]]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1D", periods=len(data)),
        data=data,
        columns=["mid"],
    )
    return backlight.datasource.from_dataframe(df, symbol, currency_unit)


@pytest.fixture
def trades(symbol, currency_unit):
    data = [1.0, -2.0, 1.0, 2.0, -4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
    index = pd.date_range(start="2018-06-06", freq="1D", periods=len(data))
    trades = []
    for i in range(0, len(data), 2):
        trade = pd.Series(index=index[i : i + 2], data=data[i : i + 2], name="amount")
        trades.append(trade)
    trades = tr.make_trades(symbol, trades, currency_unit)
    return trades


@pytest.fixture
def closed_trades(symbol, currency_unit):
    data = [1.0, -1.0, 2.0, -2.0, -4.0, 4.0, 1.0, -1.0, 1.0, -1.0]
    index = pd.date_range(start="2018-06-06", freq="1D", periods=len(data))
    trades = []
    for i in range(0, len(data), 2):
        trade = pd.Series(index=index[i : i + 2], data=data[i : i + 2], name="amount")
        trades.append(trade)
    trades = module.make_trades(symbol, trades, currency_unit)
    return trades


def test__calculate_pl(symbol, currency_unit):
    periods = 3
    dates = pd.date_range(start="2018-12-01", periods=periods)
    amounts = [1.0, -1.0]

    t00 = tr.Transaction(timestamp=dates[0], amount=amounts[0])
    t11 = tr.Transaction(timestamp=dates[1], amount=amounts[1])
    t10 = tr.Transaction(timestamp=dates[1], amount=amounts[0])
    t01 = tr.Transaction(timestamp=dates[0], amount=amounts[1])
    t20 = tr.Transaction(timestamp=dates[2], amount=amounts[0])

    mkt = backlight.datasource.from_dataframe(
        pd.DataFrame(index=dates, data=[[0], [1], [2]], columns=["mid"]),
        symbol,
        currency_unit,
    )

    trade = tr.make_trade([t00, t11])
    assert module._calculate_pl(trade, mkt) == 1.0

    trade = tr.make_trade([t00, t01])
    assert module._calculate_pl(trade, mkt) == 0.0

    trade = tr.make_trade([t11, t20])
    assert module._calculate_pl(trade, mkt) == -1.0

    trade = tr.make_trade([t00, t10, t20])
    assert module._calculate_pl(trade, mkt) == 3.0


def test_calculate_trade_performance(trades, market):
    principal = 100.0
    metrics = module.calculate_trade_performance(trades, market, principal=principal)

    expected_cnt_trade = 5
    expected_cnt_win = 3
    expected_cnt_lose = 1
    expected_total_pl = 2.0
    expected_win_pl = 5.0
    expected_lose_pl = -3.0
    expected_trade_amount = 14.0
    expected_sharpe = 2.9452967928116256
    expected_avg_pl = expected_total_pl / expected_trade_amount
    expected_max_drawdown = 2.0
    expected_avg_win_pl = expected_win_pl / expected_cnt_win
    expected_avg_lose_pl = expected_lose_pl / expected_cnt_lose
    expected_avg_pl_per_trade = expected_total_pl / expected_cnt_trade
    expected_win_ratio = expected_cnt_win / expected_cnt_trade
    expected_lose_ratio = expected_cnt_lose / expected_cnt_trade
    assert metrics.loc["metrics", "cnt_trade"] == expected_cnt_trade
    assert metrics.loc["metrics", "cnt_win"] == expected_cnt_win
    assert metrics.loc["metrics", "cnt_lose"] == expected_cnt_lose
    assert metrics.loc["metrics", "total_win_pl"] == expected_win_pl
    assert metrics.loc["metrics", "total_win_pl"] == expected_win_pl
    assert metrics.loc["metrics", "total_win_pl"] == expected_win_pl
    assert metrics.loc["metrics", "total_lose_pl"] == expected_lose_pl
    assert metrics.loc["metrics", "cnt_amount"] == expected_trade_amount
    assert metrics.loc["metrics", "avg_pl_per_amount"] == expected_avg_pl
    assert metrics.loc["metrics", "sharpe"] == expected_sharpe
    assert metrics.loc["metrics", "sharpe"] == expected_sharpe
    assert metrics.loc["metrics", "max_drawdown"] == expected_max_drawdown
    assert metrics.loc["metrics", "avg_win_pl"] == expected_avg_win_pl
    assert metrics.loc["metrics", "avg_lose_pl"] == expected_avg_lose_pl
    assert metrics.loc["metrics", "avg_pl_per_trade"] == expected_avg_pl_per_trade
    assert metrics.loc["metrics", "win_ratio"] == expected_win_ratio
    assert metrics.loc["metrics", "lose_ratio"] == expected_lose_ratio


def test_calculate_trade_performance_with_closed_trades(closed_trades, market):
    principal = 100.0
    metrics = module.calculate_trade_performance(
        closed_trades, market, principal=principal
    )

    expected_cnt_trade = 5
    expected_cnt_win = 3
    expected_cnt_lose = 1
    expected_total_pl = 0
    expected_win_pl = 4.0
    expected_lose_pl = -4.0
    expected_trade_amount = 18
    expected_sharpe = 1.1634840542100998e-14
    expected_avg_pl = expected_total_pl / expected_trade_amount
    expected_max_drawdown = 4.0
    expected_avg_win_pl = expected_win_pl / expected_cnt_win
    expected_avg_lose_pl = expected_lose_pl / expected_cnt_lose
    expected_avg_pl_per_trade = expected_total_pl / expected_cnt_trade
    expected_win_ratio = expected_cnt_win / expected_cnt_trade
    expected_lose_ratio = expected_cnt_lose / expected_cnt_trade
    assert metrics.loc["metrics", "cnt_trade"] == expected_cnt_trade
    assert metrics.loc["metrics", "cnt_win"] == expected_cnt_win
    assert metrics.loc["metrics", "cnt_lose"] == expected_cnt_lose
    assert metrics.loc["metrics", "total_win_pl"] == expected_win_pl
    assert metrics.loc["metrics", "total_win_pl"] == expected_win_pl
    assert metrics.loc["metrics", "total_win_pl"] == expected_win_pl
    assert metrics.loc["metrics", "total_lose_pl"] == expected_lose_pl
    assert metrics.loc["metrics", "cnt_amount"] == expected_trade_amount
    assert metrics.loc["metrics", "avg_pl_per_amount"] == expected_avg_pl
    assert metrics.loc["metrics", "sharpe"] == expected_sharpe
    assert metrics.loc["metrics", "max_drawdown"] == expected_max_drawdown
    assert metrics.loc["metrics", "avg_win_pl"] == expected_avg_win_pl
    assert metrics.loc["metrics", "avg_lose_pl"] == expected_avg_lose_pl
    assert metrics.loc["metrics", "avg_pl_per_trade"] == expected_avg_pl_per_trade
    assert metrics.loc["metrics", "win_ratio"] == expected_win_ratio
    assert metrics.loc["metrics", "lose_ratio"] == expected_lose_ratio


@pytest.mark.parametrize(
    "a_trades, total_count, win_count, lose_count",
    [[trades, 5, 3, 1], [closed_trades, 5, 3, 1]],
)
def test_count_trades(a_trades, total_count, win_count, lose_count, market):
    assert (total_count, win_count, lose_count) == module.count_trades(
        a_trades(symbol(), currency_unit()), market
    )


def test_calculate_trade_performance_for_sametime_trade_case():
    symbol = "USDJPY"
    data = [[1.0], [2.0], [3.0]]
    df = pd.DataFrame(
        index=pd.date_range(start="2019-06-06", freq="3S", periods=len(data)),
        data=data,
        columns=["mid"],
    )
    mid = backlight.datasource.from_dataframe(df, symbol, Currency.USD)

    data = [-1.0, 1.0]
    sr = pd.Series(
        index=[
            pd.Timestamp("2019-06-06 00:00:03"),
            pd.Timestamp("2019-06-06 00:00:03"),
        ],
        data=data,
        name="amount",
    )
    trade = sr

    trades = make_trades(symbol, [trade], Currency.USD)

    metrics = module.calculate_trade_performance(trades, mid)

    expected_cnt_trade = 1
    expected_cnt_win = 0
    expected_cnt_lose = 0
    expected_total_pl = 0
    expected_win_pl = 0
    expected_lose_pl = 0
    expected_trade_amount = 0
    # expected_sharpe = nan
    expected_avg_pl = 0  # expected_total_pl / expected_trade_amount
    expected_max_drawdown = 0
    expected_avg_win_pl = 0  # expected_win_pl / expected_cnt_win
    expected_avg_lose_pl = 0  # expected_lose_pl / expected_cnt_lose
    expected_avg_pl_per_trade = expected_total_pl / expected_cnt_trade
    expected_win_ratio = expected_cnt_win / expected_cnt_trade
    expected_lose_ratio = expected_cnt_lose / expected_cnt_trade
    assert metrics.loc["metrics", "cnt_trade"] == expected_cnt_trade
    assert metrics.loc["metrics", "cnt_win"] == expected_cnt_win
    assert metrics.loc["metrics", "cnt_lose"] == expected_cnt_lose
    assert metrics.loc["metrics", "total_win_pl"] == expected_win_pl
    assert metrics.loc["metrics", "total_win_pl"] == expected_win_pl
    assert metrics.loc["metrics", "total_win_pl"] == expected_win_pl
    assert metrics.loc["metrics", "total_lose_pl"] == expected_lose_pl
    assert metrics.loc["metrics", "cnt_amount"] == expected_trade_amount
    assert metrics.loc["metrics", "avg_pl_per_amount"] == expected_avg_pl
    assert math.isnan(metrics.loc["metrics", "sharpe"])
    assert metrics.loc["metrics", "max_drawdown"] == expected_max_drawdown
    assert metrics.loc["metrics", "avg_win_pl"] == expected_avg_win_pl
    assert metrics.loc["metrics", "avg_lose_pl"] == expected_avg_lose_pl
    assert metrics.loc["metrics", "avg_pl_per_trade"] == expected_avg_pl_per_trade
    assert metrics.loc["metrics", "win_ratio"] == expected_win_ratio
    assert metrics.loc["metrics", "lose_ratio"] == expected_lose_ratio
