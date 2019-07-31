from backlight.positions import positions as module
import pandas as pd
import pytest

import backlight.datasource
import backlight.positions
from backlight.trades.trades import make_trades
from backlight.asset.currency import Currency


@pytest.fixture
def symbol():
    return "usdjpy"


@pytest.fixture
def currency_unit():
    return Currency.JPY


@pytest.fixture
def mid(symbol, currency_unit):
    data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [9.0]]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=len(data)),
        data=data,
        columns=["mid"],
    )
    return backlight.datasource.from_dataframe(df, symbol, currency_unit)


@pytest.fixture
def askbid(symbol, currency_unit):
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
        [9.5, 8.5],
    ]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=len(data)),
        data=data,
        columns=["ask", "bid"],
    )
    return backlight.datasource.from_dataframe(df, symbol, currency_unit)


@pytest.fixture
def trades(symbol, currency_unit):
    data = [1.0, -2.0, 1.0, 2.0, -4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
    sr = pd.Series(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=len(data)),
        data=data,
        name="amount",
    )
    trade = sr
    trades = make_trades(symbol, [trade], currency_unit)
    return trades


@pytest.fixture
def positions(trades, mid):
    # positions should be
    # data = [
    #     [1.0, 1.0, -1.0],  # value = 0.0, pl = None
    #     [-1.0, 2.0, 3.0],  # value = 1.0, pl = 1.0
    #     [0.0, 3.0, 0.0],  # value = 0.0, pl = -1.0
    #     [2.0, 4.0, -8.0],  # value = 0.0, pl = 0.0
    #     [-2.0, 5.0, 12.0],  # value = 2.0, pl = 2.0
    #     [0.0, 6.0, 0.0],  # value = 0.0, pl = -2.0
    #     [1.0, 7.0, -7.0],  # value = 0.0, pl = 0.0
    #     [1.0, 8.0, -7.0],  # value = 1.0, pl = 1.0
    #     [2.0, 9.0, -16.0],  # value = 2.0, pl = 1.0
    #     [2.0, 9.0, -16.0],  # value = 2.0, pl = 0.0
    # ]
    # columns = ["amount", "price", "principal"]
    return backlight.positions.calculate_positions(trades, mid)


def test_calculate_positions(trades, mid):
    data = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, -1.0],
        [-1.0, 2.0, 3.0],
        [0.0, 3.0, 0.0],
        [2.0, 4.0, -8.0],
        [-2.0, 5.0, 12.0],
        [0.0, 6.0, 0.0],
        [1.0, 7.0, -7.0],
        [1.0, 8.0, -7.0],
        [2.0, 9.0, -16.0],
        [2.0, 9.0, -16.0],
    ]
    df = pd.DataFrame(
        index=pd.date_range(
            start="2018-06-05 23:59:00", freq="1min", periods=len(data)
        ),
        data=data,
        columns=["amount", "price", "principal"],
    )
    expected = module.Positions(df)
    positions = module.calculate_positions(trades, mid)
    pd.testing.assert_frame_equal(positions, expected)


def test_calculate_positions_with_askbid(trades, askbid):
    data = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, -1.5],
        [-1.0, 2.0, 1.5],
        [0.0, 3.0, -2.0],
        [2.0, 4.0, -11.0],
        [-2.0, 5.0, 7.0],
        [0.0, 6.0, -6.0],
        [1.0, 7.0, -13.5],
        [1.0, 8.0, -13.5],
        [2.0, 9.0, -23.0],
        [2.0, 9.0, -23.0],
    ]
    df = pd.DataFrame(
        index=pd.date_range(
            start="2018-06-05 23:59:00", freq="1min", periods=len(data)
        ),
        data=data,
        columns=["amount", "price", "principal"],
    )
    expected = module.Positions(df)
    positions = module.calculate_positions(trades, askbid)
    pd.testing.assert_frame_equal(positions, expected)


def test_calculate_positions_bfill(trades, mid):
    data = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 2.0, 3.0],
        [-1.0, 2.0, 3.0],
        [0.0, 3.0, 0.0],
        [0.0, 3.0, 0.0],
        [2.0, 4.0, -8.0],
        [2.0, 4.0, -8.0],
        [-2.0, 5.0, 12.0],
        [-2.0, 5.0, 12.0],
        [0.0, 6.0, 0.0],
        [0.0, 6.0, 0.0],
        [1.0, 7.0, -7.0],
        [1.0, 7.0, -7.0],
        [1.0, 8.0, -7.0],
        [1.0, 8.0, -7.0],
        [2.0, 9.0, -16.0],
        [2.0, 9.0, -16.0],
        [2.0, 9.0, -16.0],
    ]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-05 23:59:30", freq="30s", periods=len(data)),
        data=data,
        columns=["amount", "price", "principal"],
    )
    expected = module.Positions(df)
    positions = module.calculate_positions(trades, mid.resample("30s").ffill())
    pd.testing.assert_frame_equal(positions, expected)
