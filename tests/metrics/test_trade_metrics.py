from backlight.metrics import trade_metrics as module
import pandas as pd
import pytest

import backlight.datasource
from backlight.trades import trades as tr


@pytest.fixture
def symbol():
    return "usdjpy"


@pytest.fixture
def market(symbol):
    data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [9.0]]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1D", periods=len(data)),
        data=data,
        columns=["mid"],
    )
    return backlight.datasource.from_dataframe(df, symbol)


@pytest.fixture
def trades(symbol):
    symbol = "usdjpy"
    data = [1.0, -2.0, 1.0, 2.0, -4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
    index = pd.date_range(start="2018-06-06", freq="1D", periods=len(data))
    trades = []
    for i in range(0, len(data), 2):
        trade = tr.from_series(
            pd.Series(index=index[i : i + 2], data=data[i : i + 2], name="amount")
        )
        trades.append(trade)
    trades = tr.from_tuple(trades, symbol)
    return trades


def test__calc_pl():
    periods = 3
    symbol = "usdjpy"
    dates = pd.date_range(start="2018-12-01", periods=periods)
    amounts = [1.0, -1.0]

    t00 = tr.Transaction(timestamp=dates[0], amount=amounts[0])
    t11 = tr.Transaction(timestamp=dates[1], amount=amounts[1])
    t10 = tr.Transaction(timestamp=dates[1], amount=amounts[0])
    t01 = tr.Transaction(timestamp=dates[0], amount=amounts[1])
    t20 = tr.Transaction(timestamp=dates[2], amount=amounts[0])

    mkt = backlight.datasource.from_dataframe(
        pd.DataFrame(index=dates, data=[[0], [1], [2]], columns=["mid"]), symbol
    )

    trade = tr.make_trade([t00, t11])
    assert module._calc_pl(trade, mkt) == 1.0

    trade = tr.make_trade([t00, t01])
    assert module._calc_pl(trade, mkt) == 0.0

    trade = tr.make_trade([t11, t20])
    assert module._calc_pl(trade, mkt) == -1.0

    trade = tr.make_trade([t00, t10, t20])
    assert module._calc_pl(trade, mkt) == 3.0


def test_calc_trade_performance(trades, market):
    principal = 100.0
    metrics = module.calc_trade_performance(trades, market, principal=principal)

    expected_cnt_trade = 5
    expected_cnt_win = 3
    expected_cnt_lose = 1
    expected_total_pl = 2.0
    expected_win_pl = 5.0
    expected_lose_pl = -3.0
    expected_trade_amount = 14.0
    expected_sharpe = 2.9452967928116256
    expected_avg_pl = expected_total_pl / expected_trade_amount
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


def test_count_trades(trades, market):
    assert (5, 3, 1) == module.count_trades(trades, market)
