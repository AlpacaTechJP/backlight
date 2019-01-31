from backlight.trades import trades as module

import pytest

import pandas as pd
import backlight.datasource


def _make_trade(transactions, symbol="hoge"):
    trade = module.Trade(symbol)
    for t in transactions:
        trade.add(t)
    return trade


@pytest.fixture
def symbol():
    return "usdjpy"


@pytest.fixture
def trades(symbol):
    data = [1.0, -2.0, 1.0, 2.0, -4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
    index = pd.date_range(start="2018-06-06", freq="1min", periods=len(data))
    trades = []
    for i in range(0, len(data), 2):
        trade = module.from_series(
            pd.Series(index=index[i : i + 2], data=data[i : i + 2], name="amount"),
            symbol,
        )
        trades.append(trade)
    trades = module.from_tuple(trades)
    return trades


@pytest.fixture
def market(symbol):
    data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [9.0]]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=len(data)),
        data=data,
        columns=["mid"],
    )
    return backlight.datasource.from_dataframe(df, symbol)


def test_Trade():
    periods = 2
    dates = pd.date_range(start="2018-12-01", periods=periods)
    amounts = range(periods)

    t00 = module.Transaction(timestamp=dates[0], amount=amounts[0])
    t11 = module.Transaction(timestamp=dates[1], amount=amounts[1])
    t01 = module.Transaction(timestamp=dates[0], amount=amounts[1])

    trade = _make_trade([t00, t11])
    expected = pd.Series(index=dates, data=amounts[:2], name="amount")
    assert (trade.amount == expected).all()

    trade = _make_trade([t00, t01])
    expected = pd.Series(
        index=[dates[0]], data=[amounts[0] + amounts[1]], name="amount"
    )
    assert (trade.amount == expected).all()

    trade = _make_trade([t11, t01, t00])
    expected = pd.Series(
        index=dates, data=[amounts[0] + amounts[1], amounts[1]], name="amount"
    )
    assert (trade.amount == expected).all()


def test_flatten(symbol, trades):
    data = [1.0, -2.0, 1.0, 2.0, -4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
    index = pd.date_range(start="2018-06-06", freq="1min", periods=len(data))
    expected = module.from_series(
        pd.Series(index=index, data=data, name="amount"), symbol
    )
    trade = module.flatten(trades)
    pd.testing.assert_series_equal(trade.amount, expected.amount)
