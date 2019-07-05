import pytest
import pandas as pd
import numpy as np
from backlight.portfolio.portfolio import construct_portfolio as module

import backlight

from backlight.trades.trades import make_trades
from backlight.asset.currency import Currency


@pytest.fixture
def trades():
    trades = []
    index = [
        "2018-06-06 00:00:00",
        "2018-06-06 00:01:00",
        "2018-06-06 00:02:00",
        "2018-06-06 00:03:00",
        "2018-06-06 00:03:00",
        "2018-06-06 00:04:00 ",
        "2018-06-06 00:05:00",
        "2018-06-06 00:05:00",
        "2018-06-06 00:06:00 ",
        "2018-06-06 00:06:00 ",
        "2018-06-06 00:07:00 ",
        "2018-06-06 00:08:00 ",
        "2018-06-06 00:09:00 ",
        "2018-06-06 00:09:00 ",
    ]

    trade = pd.Series(
        index=pd.to_datetime(index),
        data=[1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1],
        name="amount",
    )
    ids = [0, 1, 0, 1, 2, 3, 2, 4, 3, 5, 4, 5, 6, 6]

    trades.append(make_trades("usdjpy", [trade], [ids]))
    trades.append(make_trades("eurjpy", [trade], [ids]))
    return trades


@pytest.fixture
def markets():
    markets = []
    symbol = "usdjpy"
    currency_unit = Currency.JPY
    periods = 10
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=np.arange(periods)[:, None],
        columns=["mid"],
    )
    markets.append(backlight.datasource.from_dataframe(df, symbol, currency_unit))

    symbol = "eurjpy"
    currency_unit = Currency.JPY
    periods = 10
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=10 - np.arange(periods)[:, None],
        columns=["mid"],
    )
    markets.append(backlight.datasource.from_dataframe(df, symbol, currency_unit))
    return markets


@pytest.fixture
def principal():
    return [10, 10]


@pytest.fixture
def lot_size():
    return [2, 2]


def test_construct_portfolio(trades, markets, principal, lot_size):
    portfolio = module(trades, markets, principal, lot_size)

    index = [
        "2018-06-05 23:59:00",
        "2018-06-06 00:00:00",
        "2018-06-06 00:01:00",
        "2018-06-06 00:02:00",
        "2018-06-06 00:03:00",
        "2018-06-06 00:04:00 ",
        "2018-06-06 00:05:00",
        "2018-06-06 00:06:00 ",
        "2018-06-06 00:07:00 ",
        "2018-06-06 00:08:00 ",
        "2018-06-06 00:09:00 ",
    ]

    data1 = [
        [0.0, 0.0, 10.0],
        [2.0, 0.0, 10.0],
        [0.0, 1.0, 12.0],
        [-2.0, 2.0, 16.0],
        [2.0, 3.0, 4.0],
        [4.0, 4.0, -4.0],
        [0.0, 5.0, 16.0],
        [-4.0, 6.0, 40.0],
        [-2.0, 7.0, 26.0],
        [0.0, 8.0, 10.0],
        [0.0, 9.0, 10.0],
    ]

    data2 = [
        [0.0, 0.0, 10.0],
        [2.0, 10.0, -10.0],
        [0.0, 9.0, 8.0],
        [-2.0, 8.0, 24.0],
        [2.0, 7.0, -4.0],
        [4.0, 6.0, -16.0],
        [0.0, 5.0, 4.0],
        [-4.0, 4.0, 20.0],
        [-2.0, 3.0, 14.0],
        [0.0, 2.0, 10.0],
        [0.0, 1.0, 10.0],
    ]

    data = [data1, data2]

    for (position, d) in zip(portfolio._positions, data):

        expected = pd.DataFrame(
            index=pd.to_datetime(index),
            data=d,
            columns=["amount", "price", "principal"],
        )
        assert ((expected == position).all()).all()
