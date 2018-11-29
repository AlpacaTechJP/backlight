from backlight.strategies import amount_based as module
import pytest
import pandas as pd
import numpy as np

import backlight
from backlight.labelizer.common import TernaryDirection
from backlight.strategies.common import Action


@pytest.fixture
def signal():
    symbol = "usdjpy"
    periods = 22
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
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ],
        columns=["up", "neutral", "down"],
    )
    signal = backlight.signal.from_dataframe(df, symbol)
    return signal


@pytest.fixture
def market():
    symbol = "usdjpy"
    periods = 22
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=np.arange(periods)[:, None],
        columns=["mid"],
    )
    market = backlight.datasource.from_dataframe(df, symbol)
    return market


def test_entry_exit_trades(market, signal):
    max_holding_time = pd.Timedelta("3min")
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    trades = module.entry_exit_trades(
        market, signal, direction_action_dict, max_holding_time
    )
    print(trades)
    expected = pd.Series(
        index=market.index,
        data=[
            1.0,  # U
            -1.0,  # D
            0.0,  # N
            0.0,  # U + D
            2.0,  # U + U
            -1.0,  # D + N
            -2.0,  # D + D
            -1.0,  # N + D
            1.0,  # N + U
            2.0,  # U + U
            1.0,  # U + N
            1.0,  # U + N
            -2.0,  # D + D
            -2.0,  # D + D
            -2.0,  # D + D
            1.0,  # N + U
            1.0,  # N + U
            1.0,  # N + U
            1.0,  # U + N
            1.0,  # U + N
            1.0,  # U + N
            -3.0,  # U + 4*N because of market close
        ],
        name="amount",
    )
    assert (trades.amount == expected).all()
