from backlight.metrics import pl as module
import pandas as pd
import pytest

from backlight.positions.positions import Positions


@pytest.fixture
def positions():
    symbol = "usdjpy"
    data = [
        [1.0, 1.0],  # pl = None
        [-1.0, 2.0],  # pl = 1.0 * (2.0 - 1.0) = 1.0
        [0.0, 3.0],  # pl = -1.0 * (3.0 - 2.0) = -1.0
        [2.0, 4.0],  # pl = 0.0 * (4.0 - 3.0) = 0.0
        [-2.0, 5.0],  # pl = 2.0 * (5.0 - 4.0) = 2.0
        [0.0, 6.0],  # pl = -2.0 * (6.0 - 5.0) = -2.0
        [1.0, 7.0],  # pl = 0.0 * (7.0 - 6.0) = 0.0
        [1.0, 8.0],  # pl = 1.0 * (8.0 - 7.0) = 1.0
        [2.0, 9.0],  # pl = 1.0 * (9.0 - 8.0) = 1.0
        [2.0, 9.0],  # pl = 2.0 * (9.0 - 9.0) = 0.0
    ]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=len(data)),
        data=data,
        columns=["amount", "price"],
    )
    positions = Positions(df)
    positions.symbol = symbol
    return positions


def test__pl(positions):
    expected = pd.Series(
        data=[1.0, -1.0, 0.0, 2.0, -2.0, 0.0, 1.0, 1.0, 0.0],
        index=positions.index[1:],
        name="pl",
    )
    assert (module._pl(positions) == expected).all()


def test__trade_amount(positions):
    expected = 13.0
    assert module._trade_amount(positions.amount) == expected


def test_calc_position_performance(positions):
    metrics = module.calc_position_performance(positions)
    expected_total_pl = 2.0
    expected_trade_amount = 13.0
    expected_avg_pl = expected_total_pl / expected_trade_amount
    assert metrics.loc["metrics", "total_pl"] == expected_total_pl
    assert metrics.loc["metrics", "trade_amount"] == expected_trade_amount
    assert metrics.loc["metrics", "avg_pl"] == expected_avg_pl
