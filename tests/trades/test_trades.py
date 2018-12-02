from backlight.trades import trades as module

import pytest

import pandas as pd
import backlight.datasource


@pytest.fixture
def trades():
    symbol = "usdjpy"
    data = [1.0, -2.0, 1.0, 2.0, -4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
    index = pd.date_range(start="2018-06-06", freq="1min", periods=len(data))
    trades = []
    for i in range(0, len(data), 2):
        trade = module.Trade(
            pd.Series(index=index[i : i + 2], data=data[i : i + 2], name="amount")
        )
        trade.symbol = symbol
        trades.append(trade)
    return trades


def test_Trade():
    periods = 2
    dates = pd.date_range(start="2018-12-01", periods=periods)
    amounts = range(periods)

    t00 = module.Transaction(timestamp=dates[0], amount=amounts[0])
    t11 = module.Transaction(timestamp=dates[1], amount=amounts[1])
    t01 = module.Transaction(timestamp=dates[0], amount=amounts[1])

    trade = module.Trade().add(t00).add(t11)
    expected = pd.Series(index=dates, data=amounts[:2], name="amount")
    assert (trade.amount == expected).all()

    trade = module.Trade().add(t00).add(t01)
    expected = pd.Series(
        index=[dates[0]], data=[amounts[0] + amounts[1]], name="amount"
    )
    assert (trade.amount == expected).all()

    trade = module.Trade().add(t11).add(t01).add(t00)
    expected = pd.Series(
        index=dates, data=[amounts[0] + amounts[1], amounts[1]], name="amount"
    )
    assert (trade.amount == expected).all()


def test__evaluate_pl():
    periods = 3
    symbol = "usdjpy"
    dates = pd.date_range(start="2018-12-01", periods=periods)
    amounts = [1.0, -1.0]

    t00 = module.Transaction(timestamp=dates[0], amount=amounts[0])
    t11 = module.Transaction(timestamp=dates[1], amount=amounts[1])
    t10 = module.Transaction(timestamp=dates[1], amount=amounts[0])
    t01 = module.Transaction(timestamp=dates[0], amount=amounts[1])
    t20 = module.Transaction(timestamp=dates[2], amount=amounts[0])

    mkt = backlight.datasource.from_dataframe(
        pd.DataFrame(index=dates, data=[[0], [1], [2]], columns=["mid"]), symbol
    )

    trade = module.make_trade(symbol).add(t00).add(t11)
    assert module._evaluate_pl(trade, mkt) == 1.0

    trade = module.make_trade(symbol).add(t00).add(t01)
    assert module._evaluate_pl(trade, mkt) == 0.0

    trade = module.make_trade(symbol).add(t11).add(t20)
    assert module._evaluate_pl(trade, mkt) == -1.0

    trade = module.make_trade(symbol).add(t00).add(t10).add(t20)
    assert module._evaluate_pl(trade, mkt) == 3.0


def test_flatten(trades):
    symbol = "usdjpy"
    data = [1.0, -2.0, 1.0, 2.0, -4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
    index = pd.date_range(start="2018-06-06", freq="1min", periods=len(data))
    expected = module._make_trade(
        pd.Series(index=index, data=data, name="amount"), symbol
    )
    trade = module.flatten(trades)
    assert (trade == expected).all()
