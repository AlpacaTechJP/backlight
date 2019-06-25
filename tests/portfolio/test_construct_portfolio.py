import pytest
import pandas as pd
import numpy as np
from backlight.portfolio.portfolio import calculate_pl as module
from backlight.positions.positions import Positions
from backlight.portfolio.portfolio import Portfolio

import backlight


@pytest.fixture
def markets():
    markets = []
    symbol = "usdjpy"
    periods = 4
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-05 23:59:00", freq="1min", periods=periods),
        data=np.array([[0, 0, 1, 2], [1, 1, 2, 4]]).T,
        columns=["bid", "ask"],
    )
    markets.append(backlight.datasource.from_dataframe(df, symbol))

    return markets


@pytest.fixture
def portfolio():
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


def test_calculate_pl():
    calculated_portfolio = module(portfolio, markets, base_ccy="usd")

    index = ["2018-06-06 00:00:00", "2018-06-06 00:01:00", "2018-06-06 00:02:00"]

    data1 = [[0.0, 1.0, 2.0], [1.5, 2.0, 2.5], [1.5, 1.75, 2.0]]

    data2 = [[0.0, 1.0, 2.0], [1.5, 2.0, 2.5], [1.5, 1.75, 2.0]]

    data3 = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]

    data = [data1, data2, data3]

    for (position, d) in zip(calculated_portfolio._positions, data):

        expected = pd.DataFrame(
            index=pd.to_datetime(index),
            data=d,
            columns=["amount", "price", "principal"],
        )
        assert ((expected == position).all()).all()
