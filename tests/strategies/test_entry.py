from backlight.strategies import amount_based as module
import pytest
import pandas as pd
import numpy as np

import backlight
from backlight.labelizer.common import TernaryDirection
from backlight.strategies.common import Action
from backlight.asset.currency import Currency


@pytest.fixture
def symbol():
    return "usdjpy"


@pytest.fixture
def currency_unit():
    return Currency.JPY


@pytest.fixture
def signal(symbol, currency_unit):
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
    signal = backlight.signal.from_dataframe(df, symbol, currency_unit)
    return signal


@pytest.fixture
def market(symbol, currency_unit):
    periods = 22
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=np.arange(periods)[:, None],
        columns=["mid"],
    )
    market = backlight.datasource.from_dataframe(df, symbol, currency_unit)
    return market


def test_direction_based_entry(market, signal):
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    trades = module.direction_based_entry(market, signal, direction_action_dict)
    expected = pd.Series(
        index=signal[signal.pred != TernaryDirection.NEUTRAL.value].index,
        data=[
            1.0,  # U
            -1.0,  # D
            1.0,  # U
            1.0,  # U
            -1.0,  # D
            -1.0,  # D
            1.0,  # U
            1.0,  # U
            1.0,  # U
            -1.0,  # D
            -1.0,  # D
            -1.0,  # D
            1.0,  # U
            1.0,  # U
            1.0,  # U
            1.0,  # U
        ],
        name="amount",
    )
    assert (trades.amount == expected).all()
