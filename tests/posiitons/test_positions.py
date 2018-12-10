from backlight.positions import positions as module
import pandas as pd
import pytest

import backlight.datasource
import backlight.positions
from backlight.trades.trades import from_series


@pytest.fixture
def symbol():
    return "usdjpy"


@pytest.fixture
def mid(symbol):
    data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [9.0]]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=len(data)),
        data=data,
        columns=["mid"],
    )
    return backlight.datasource.from_dataframe(df, symbol)


@pytest.fixture
def askbid(symbol):
    data = [
        [1.5, 0.5],
        [2.5, 1.5],
        [3.5, 2.5],
        [4.5, 3.5],
        [5.5, 4.5],
        [6.5, 5.5],
        [7.5, 6.5],
        [8.5, 7.5],
        [9.5, 8.5],
        [9.5, 8.5]
    ]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=len(data)),
        data=data,
        columns=["ask", "bid"],
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
    trade = from_series(sr, symbol)
    return (trade,)


@pytest.fixture
def positions(trades, mid):
    # positions should be
    # data = [
    #     [1.0, 1.0],  # pl = None
    #     [-1.0, 2.0],  # pl = 1.0 * (2.0 - 1.0) = 1.0
    #     [0.0, 3.0],  # pl = -1.0 * (3.0 - 2.0) = -1.0
    #     [2.0, 4.0],  # pl = 0.0 * (4.0 - 3.0) = 0.0
    #     [-2.0, 5.0],  # pl = 2.0 * (5.0 - 4.0) = 2.0
    #     [0.0, 6.0],  # pl = -2.0 * (6.0 - 5.0) = -2.0
    #     [1.0, 7.0],  # pl = 0.0 * (7.0 - 6.0) = 0.0
    #     [1.0, 8.0],  # pl = 1.0 * (8.0 - 7.0) = 1.0
    #     [2.0, 9.0],  # pl = 1.0 * (9.0 - 8.0) = 1.0
    #     [2.0, 9.0],  # pl = 2.0 * (9.0 - 9.0) = 0.0
    # ]
    return backlight.positions.calc_positions(trades, mid)


def test_calc_positions(trades, mid):
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
    positions = module.calc_positions(trades, mid)
    pd.testing.assert_frame_equal(positions, expected)


def test_calc_positions_bfill(trades, mid):
    data = [
        [1.0, 1.0],
        [1.0, 1.0],
        [-1.0, 2.0],
        [-1.0, 2.0],
        [0.0, 3.0],
        [0.0, 3.0],
        [2.0, 4.0],
        [2.0, 4.0],
        [-2.0, 5.0],
        [-2.0, 5.0],
        [0.0, 6.0],
        [0.0, 6.0],
        [1.0, 7.0],
        [1.0, 7.0],
        [1.0, 8.0],
        [1.0, 8.0],
        [2.0, 9.0],
        [2.0, 9.0],
        [2.0, 9.0],
    ]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="30s", periods=len(data)),
        data=data,
        columns=["amount", "price"],
    )
    expected = module.Positions(df)
    positions = module.calc_positions(trades, mid.resample("30s").ffill())
    pd.testing.assert_frame_equal(positions, expected)


def test_calc_pl(positions):
    expected = pd.Series(
        data=[1.0, -1.0, 0.0, 2.0, -2.0, 0.0, 1.0, 1.0, 0.0],
        index=positions.index[1:],
        name="pl",
    )
    assert (module.calc_pl(positions) == expected).all()
