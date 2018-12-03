from backlight.positions import positions as module
import pandas as pd
import pytest

import backlight.datasource
from backlight.trades.trades import _make_trade


@pytest.fixture
def symbol():
    return "usdjpy"


@pytest.fixture
def market(symbol):
    data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [9.0]]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=len(data)),
        data=data,
        columns=["mid"],
    )
    return backlight.datasource.from_dataframe(df, symbol)


@pytest.fixture
def trades(symbol):
    data = [1.0, -2.0, 1.0, 2.0, -4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
    sr = pd.Series(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=len(data)),
        data=data,
        name="amount",
    )
    trade = _make_trade(sr, symbol)
    return [trade]


def test_calc_positions(trades, market):
    data = [
        [1.0, 1.0],
        [-1.0, 2.0],
        [0.0, 3.0],
        [2.0, 4.0],
        [-2.0, 5.0],
        [0.0, 6.0],
        [1.0, 7.0],
        [1.0, 8.0],
        [2.0, 9.0],
        [2.0, 9.0],
    ]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=len(data)),
        data=data,
        columns=["amount", "price"],
    )
    expected = module.Positions(df)
    positions = module.calc_positions(trades, market)
    pd.testing.assert_frame_equal(positions, expected)
