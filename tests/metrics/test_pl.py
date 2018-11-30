from backlight.metrics import pl as module
import pandas as pd
import pytest

from backlight.positions.positions import Positions


@pytest.fixture
def positions():
    symbol = "usdjpy"
    periods = 22
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=[
            [1, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
        ],
        columns=["amount", "price"],
    )
    positions = Positions(df)
    positions.symbol = symbol
    return positions


def test_calc_position_performance(positions):
    pass
