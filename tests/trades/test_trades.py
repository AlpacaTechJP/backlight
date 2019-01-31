from backlight.trades import trades as module

import pytest

import pandas as pd
import backlight.datasource


@pytest.fixture
def symbol():
    return "usdjpy"


@pytest.fixture
def trades(symbol):
    data = [1.0, -2.0, 1.0, 2.0, -4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
    index = pd.date_range(start="2018-06-06", freq="1min", periods=len(data))
    trades = []
    for i in range(0, len(data), 2):
        trade = module.Trade(
            index=index[i : i + 2], data=data[i : i + 2], name="amount"
        )
        trades.append(trade)
    trades = module.from_tuple(trades, symbol)
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

    trade = module.make_trade([t00, t11])
    expected = pd.Series(index=dates, data=amounts[:2], name="amount")
    assert (trade == expected).all()

    trade = module.make_trade([t00, t01])
    expected = pd.Series(
        index=[dates[0]], data=[amounts[0] + amounts[1]], name="amount"
    )
    assert (trade == expected).all()

    trade = module.make_trade([t11, t01, t00])
    expected = pd.Series(
        index=dates, data=[amounts[0] + amounts[1], amounts[1]], name="amount"
    )
    assert (trade == expected).all()
