from backlight.trades import trades as module

import pytest

import pandas as pd


@pytest.fixture
def symbol():
    return "usdjpy"


@pytest.fixture
def trades(symbol):
    data = [1.0, -2.0, 1.0, 2.0, -4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
    index = pd.date_range(start="2018-06-06", freq="1min", periods=len(data))
    trades = []
    for i in range(0, len(data), 2):
        trade = pd.Series(index=index[i : i + 2], data=data[i : i + 2], name="amount")
        trades.append(trade)
    trades = module.make_trades(symbol, trades)
    return trades


def test_trades_ids(trades):
    expected = [0, 1, 2, 3, 4]
    assert trades.ids == expected


def test_trades_amount(trades):
    data = [1.0, -2.0, 1.0, 2.0, -4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
    index = pd.date_range(start="2018-06-06", freq="1min", periods=len(data))
    expected = pd.Series(data=data, index=index, name="amount")
    pd.testing.assert_series_equal(trades.amount, expected)


def test_trades_filter_trade(trades):
    data = [1.0, -2.0]
    index = pd.date_range(start="2018-06-06", freq="1min", periods=len(data))
    expected = pd.Series(data=data, index=index, name="amount")
    result = trades.filter_trade(trades.index == index[0])
    pd.testing.assert_series_equal(result.amount, expected)


def test_trades_get_trade(trades):
    data = [1.0, -2.0]
    index = pd.date_range(start="2018-06-06", freq="1min", periods=len(data))
    expected = pd.Series(data=data, index=index, name="amount")
    pd.testing.assert_series_equal(trades.get_trade(0), expected)


def test_trades_add_trade(trades):
    data = [1.0, -2.0]
    trade_id = 9
    index = pd.date_range(start="2018-06-06", freq="1min", periods=len(data))
    t = pd.Series(data=data, index=index, name="amount")

    expected = t
    trades = trades.add_trade(expected, trade_id)
    pd.testing.assert_series_equal(trades.get_trade(trade_id), expected)


def test_make_trade():
    periods = 2
    dates = pd.date_range(start="2018-12-01", periods=periods)
    amounts = range(periods)

    t00 = module.Transaction(timestamp=dates[0], amount=amounts[0])
    t11 = module.Transaction(timestamp=dates[1], amount=amounts[1])
    t01 = module.Transaction(timestamp=dates[0], amount=amounts[1])

    trade = module.make_trade([t00, t11])
    expected = pd.Series(index=dates, data=amounts[:2], name="amount")
    pd.testing.assert_series_equal(trade, expected)

    trade = module.make_trade([t00, t01])
    expected = pd.Series(
        index=[dates[0]], data=[amounts[0] + amounts[1]], name="amount"
    )
    pd.testing.assert_series_equal(trade, expected)

    trade = module.make_trade([t11, t01, t00])
    expected = pd.Series(
        index=dates, data=[amounts[0] + amounts[1], amounts[1]], name="amount"
    )
    pd.testing.assert_series_equal(trade, expected)
