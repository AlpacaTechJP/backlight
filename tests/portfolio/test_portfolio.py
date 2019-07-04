import pytest
import pandas as pd
import numpy as np
from backlight.portfolio.portfolio import construct_portfolio as module
from backlight.portfolio.portfolio import _fusion_positions
import backlight.positions.positions


import backlight
from backlight.portfolio.portfolio import construct_portfolio as module
from backlight.trades.trades import make_trades
from backlight.positions.positions import Positions
from backlight.portfolio.portfolio import Portfolio, calculate_pl


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
    trades.append(make_trades("usdjpy", [trade], [ids]))
    return trades


@pytest.fixture
def markets():
    markets = []
    symbol = "usdjpy"
    periods = 10
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=np.arange(periods)[:, None],
        columns=["mid"],
    )
    markets.append(backlight.datasource.from_dataframe(df, symbol))

    symbol = "eurjpy"
    periods = 10
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=10 - np.arange(periods)[:, None],
        columns=["mid"],
    )
    markets.append(backlight.datasource.from_dataframe(df, symbol))

    symbol = "usdjpy"
    periods = 10
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=np.arange(periods)[:, None],
        columns=["mid"],
    )
    markets.append(backlight.datasource.from_dataframe(df, symbol))
    return markets


@pytest.fixture
def principal():
    return [10, 10, 10]


@pytest.fixture
def lot_size():
    return [2, 2, 2]


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

    data2 = [
        [0.0, 0.0, 20.0],
        [4.0, 0.0, 20.0],
        [0.0, 2.0, 24.0],
        [-4.0, 4.0, 32.0],
        [4.0, 6.0, 8.0],
        [8.0, 8.0, -8.0],
        [0.0, 10.0, 32.0],
        [-8.0, 12.0, 80.0],
        [-4.0, 14.0, 52.0],
        [0.0, 16.0, 20.0],
        [0.0, 18.0, 20.0],
    ]

    data = [data1, data2]

    for (position, d) in zip(portfolio._positions, data):

        expected = pd.DataFrame(
            index=pd.to_datetime(index),
            data=d,
            columns=["amount", "price", "principal"],
        )
        assert ((expected == position).all()).all()

def test_fusion_positions():

    periods = 3
    data = np.arange(periods * 3).reshape((periods, 3))
    columns = ["amount", "price", "principal"]

    positions_list = []
    df = pd.DataFrame(
        data=data,
        index=pd.date_range("2012-1-1", periods=periods, freq="D"),
        columns=columns,
    )
    symbol = "usdjpy"
    positions_list.append(backlight.positions.positions.from_dataframe(df, symbol))

    df = pd.DataFrame(
        data=data,
        index=pd.date_range("2012-1-2", periods=periods, freq="D"),
        columns=columns,
    )
    symbol = "usdjpy"
    positions_list.append(backlight.positions.positions.from_dataframe(df, symbol))

    df = pd.DataFrame(
        data=data,
        index=pd.date_range("2012-1-4", periods=periods, freq="D"),
        columns=columns,
    )
    symbol = "eurjpy"
    positions_list.append(backlight.positions.positions.from_dataframe(df, symbol))

    fusioned = _fusion_positions(positions_list)

    data1 = np.arange(periods * 3).reshape((periods, 3))
    data2 = [[0, 1, 2], [3, 5, 7], [9, 11, 13], [6, 7, 8]]

    df1 = pd.DataFrame(
        data=data1,
        index=pd.date_range("2012-1-1", periods=periods, freq="D"),
        columns=columns,
    )
    df2 = pd.DataFrame(
        data=data2,
        index=pd.date_range("2012-1-1", periods=periods + 1, freq="D"),
        columns=columns,
    )

    expected = [df1, df2]

    for exp, fus in zip(expected, fusioned):
        assert exp.all().all() == fus.all().all()

@pytest.fixture
def mid_markets():
    markets = []
    symbol = "usdjpy"
    periods = 4
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=np.array([0, 1, 2, 4]),
        columns=["mid"],
    )
    markets.append(backlight.datasource.from_dataframe(df, symbol))
    return markets


@pytest.fixture
def simple_portfolio():
    ptf = []
    periods = 4
    for symbol in ["usdjpy", "eurjpy", "eurusd"]:
        df = pd.DataFrame(
            index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
            data=np.arange(3 * periods).reshape((periods, 3)),
            columns=["amount", "price", "principal"],
        )
        p = Positions(df)
        p.symbol = symbol
        ptf.append(p)
    return Portfolio(ptf)


def test_calculate_pl(simple_portfolio, mid_markets):
    calculated_portfolio = calculate_pl(simple_portfolio, mid_markets, base_ccy="usd")
    expected = pd.Series(
        index=["2018-06-06 00:01:00", "2018-06-06 00:02:00", "2018-06-06 00:03:00"],
        data=[45.0, 66.0, 76.5],
    )
    assert ((expected == calculated_portfolio).all()).all()

    calculated_portfolio = calculate_pl(simple_portfolio, mid_markets, base_ccy="jpy")
    expected = pd.Series(
        index=["2018-06-06 00:01:00", "2018-06-06 00:02:00", "2018-06-06 00:03:00"],
        data=[45.0, 132.0, 306.0],
    )
    assert ((expected == calculated_portfolio).all()).all()
