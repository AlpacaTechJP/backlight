from backlight.portfolio.strategy import create_simple_trades as module
import pytest
import pandas as pd
import numpy as np

import backlight
from backlight.asset.currency import Currency
from backlight.portfolio.strategy import (
    equally_weighted_portfolio,
    _calculate_principals_lot_sizes,
)
from backlight.trades.trades import make_trades


@pytest.fixture
def strategy_name():
    return "simple_entry_and_exit"


@pytest.fixture
def strategy_params():
    return {"max_holding_time": pd.Timedelta("2min")}


@pytest.fixture
def signals():
    signals = []

    symbol = "USDJPY"
    currency_unit = Currency.JPY
    periods = 10
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=[
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ],
        columns=["up", "neutral", "down"],
    )
    signals.append(backlight.signal.from_dataframe(df, symbol, currency_unit))

    symbol = "EURJPY"
    currency_unit = Currency.JPY
    periods = 10
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=[
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ],
        columns=["up", "neutral", "down"],
    )
    signals.append(backlight.signal.from_dataframe(df, symbol, currency_unit))

    return signals


@pytest.fixture
def markets():
    markets = []
    symbol = "USDJPY"
    currency_unit = Currency.JPY
    periods = 10
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=np.arange(periods)[:, None],
        columns=["mid"],
    )
    markets.append(backlight.datasource.from_dataframe(df, symbol, currency_unit))

    symbol = "EURJPY"
    currency_unit = Currency.JPY
    periods = 10
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=10 - np.arange(periods)[:, None],
        columns=["mid"],
    )
    markets.append(backlight.datasource.from_dataframe(df, symbol, currency_unit))
    return markets


def test_create_simple_trades(markets, signals, strategy_name, strategy_params):
    created_trades = module(markets, signals, strategy_name, strategy_params)

    expected = pd.DataFrame(
        index=markets[0].index,
        data=[1, -1, -1, 2, 1, -2, -2, 1, 1, 0],
        columns=["amount"],
    )

    for trades in created_trades:
        assert (trades.amount == expected.amount).all()


@pytest.fixture
def markets2():
    markets = []
    periods = 12

    for symbol, price in zip(["USDJPY", "EURJPY", "EURUSD"], [100.0, 200.0, 2.0]):
        currency_unit = Currency[symbol[-3:]]
        quote_currency = Currency[symbol[:3]]
        df = pd.DataFrame(
            index=pd.date_range(
                start="2018-06-05 23:58:00", freq="1min", periods=periods
            ),
            data=np.repeat(price, periods)[:, None]
            # + np.array(np.arange(periods) * price / (5 * periods))[:, None]
            ,
            columns=["mid"],
        )
        markets.append(
            backlight.datasource.from_dataframe(
                df, symbol, currency_unit, quote_currency=quote_currency
            )
        )

    return markets


@pytest.fixture
def trades():
    trades = []

    periods = 10

    for symbol in ["USDJPY", "EURJPY", "EURUSD"]:
        currency_unit = Currency[symbol[-3:]]
        trade = pd.Series(
            index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
            data=[1, 1, -1, -1, 1, 1, -1, 1, -1, -1],
            name="amount",
        )
        ids = [0, 1, 0, 1, 2, 3, 2, 4, 3, 4]
        trades.append(make_trades(symbol, [trade], currency_unit, [ids]))

    return trades


def test__calculate_principals_lot_sizes(trades, markets2):
    principal = 150
    max_amount = 50
    currency_unit = Currency.USD

    principals, lot_sizes = _calculate_principals_lot_sizes(
        trades, markets2, principal, max_amount, currency_unit
    )

    expected_principals = {"USDJPY": 5000.0, "EURJPY": 5000.0, "EURUSD": 50.0}
    expected_lot_sizes = {"USDJPY": 1.0, "EURJPY": 1.0, "EURUSD": 1.0}

    assert expected_principals == principals
    assert expected_lot_sizes == lot_sizes


def test_equally_weighted_portfolio(markets2, trades):
    portfolio = equally_weighted_portfolio(trades, markets2, 150, 50, Currency.USD)

    index = pd.date_range(start="2018-06-05 23:59:00", freq="1min", periods=11)

    data1 = [
        [0.0, 0.0, 50.0],
        [1.0, 1.0, 49.0],
        [2.0, 1.0, 48.0],
        [1.0, 1.0, 49.0],
        [0.0, 1.0, 50.0],
        [1.0, 1.0, 49.0],
        [2.0, 1.0, 48.0],
        [1.0, 1.0, 49.0],
        [2.0, 1.0, 48.0],
        [1.0, 1.0, 49.0],
        [0.0, 1.0, 50.0],
    ]

    data2 = [
        [0.0, 0.0, 50.0],
        [1.0, 2.0, 48.0],
        [2.0, 2.0, 46.0],
        [1.0, 2.0, 48.0],
        [0.0, 2.0, 50.0],
        [1.0, 2.0, 48.0],
        [2.0, 2.0, 46.0],
        [1.0, 2.0, 48.0],
        [2.0, 2.0, 46.0],
        [1.0, 2.0, 48.0],
        [0.0, 2.0, 50.0],
    ]

    data3 = [
        [0.0, 0.0, 50.0],
        [1.0, 2.0, 48.0],
        [2.0, 2.0, 46.0],
        [1.0, 2.0, 48.0],
        [0.0, 2.0, 50.0],
        [1.0, 2.0, 48.0],
        [2.0, 2.0, 46.0],
        [1.0, 2.0, 48.0],
        [2.0, 2.0, 46.0],
        [1.0, 2.0, 48.0],
        [0.0, 2.0, 50.0],
    ]

    data = [data1, data2, data3]

    for position, d in zip(portfolio._positions, data):

        expected = pd.DataFrame(
            index=index, data=d, columns=["amount", "price", "principal"]
        )
        assert ((expected == position).all()).all()
