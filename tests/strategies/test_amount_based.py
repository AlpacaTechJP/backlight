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
            -3.0,  # U + 4N because of market close
        ],
        name="amount",
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected).all()


def test_simple_entry(market, signal):
    max_holding_time = pd.Timedelta("3min")
    trades = module.simple_entry(market, signal, max_holding_time)
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
            -3.0,  # U + 4D because of market close
        ],
        name="amount",
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected).all()


def test_only_entry_short(market, signal):
    max_holding_time = pd.Timedelta("3min")
    trades = module.only_entry_short(market, signal, max_holding_time)
    expected = pd.Series(
        index=market.index,
        data=[
            0.0,  # N
            -1.0,  # D
            0.0,  # N
            0.0,  # N + N
            1.0,  # N + U
            -1.0,  # D + N
            -1.0,  # D + N
            0.0,  # N + N
            1.0,  # N + U
            1.0,  # N + U
            0.0,  # N + N
            0.0,  # N + N
            -1.0,  # D + N
            -1.0,  # D + N
            -1.0,  # D + N
            1.0,  # N + U
            1.0,  # N + U
            1.0,  # N + U
            0.0,  # N + N
            0.0,  # N + N
            0.0,  # N + N
            0.0,  # N + 4N because of market close
        ],
        name="amount",
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected).all()


def test_only_entry_long(market, signal):
    max_holding_time = pd.Timedelta("3min")
    trades = module.only_entry_long(market, signal, max_holding_time)
    expected = pd.Series(
        index=market.index,
        data=[
            1.0,  # U
            0.0,  # N
            0.0,  # N
            0.0,  # U + D
            1.0,  # U + N
            0.0,  # N + N
            -1.0,  # N + D
            -1.0,  # N + D
            0.0,  # N + N
            1.0,  # U + N
            1.0,  # U + N
            1.0,  # U + N
            -1.0,  # N + D
            -1.0,  # N + D
            -1.0,  # N + D
            0.0,  # N + N
            0.0,  # N + N
            0.0,  # N + N
            1.0,  # U + N
            1.0,  # U + N
            1.0,  # U + N
            -3.0,  # U + 4D because of market close
        ],
        name="amount",
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected).all()


def test_exit_opposite_signal(market, signal):
    max_holding_time = pd.Timedelta("3min")
    trades = module.exit_on_oppsite_signals(market, signal, max_holding_time)
    expected = pd.Series(
        index=market.index,
        data=[
            1.0,  # U
            -2.0,  # D + D
            0.0,  # N
            2.0,  # U + U
            1.0,  # U
            -3.0,  # D + 2D
            -1.0,  # D
            0.0,  # N
            1.0,  # N + U
            2.0,  # U + U
            1.0,  # U
            1.0,  # U
            -4.0,  # D + 3D
            -1.0,  # D
            -1.0,  # D
            1.0,  # N + U
            1.0,  # N + U
            1.0,  # N + U
            1.0,  # U
            1.0,  # U
            1.0,  # U
            -3.0,  # U + 4D because of market close
        ],
        name="amount",
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected).all()


def test_exit_other_signal(market, signal):
    max_holding_time = pd.Timedelta("3min")
    trades = module.exit_on_other_signals(market, signal, max_holding_time)
    expected = pd.Series(
        index=market.index,
        data=[
            1.0,  # U
            -2.0,  # D + D
            1.0,  # N + U
            1.0,  # U
            1.0,  # U
            -3.0,  # D + 2D
            -1.0,  # D
            2.0,  # N + 2U
            0.0,  # N
            1.0,  # U
            1.0,  # U
            1.0,  # U
            -4.0,  # D + 3D
            -1.0,  # D
            -1.0,  # D
            3.0,  # N + 3U
            0.0,  # N
            0.0,  # N
            1.0,  # U
            1.0,  # U
            1.0,  # U
            -3.0,  # U + 4D because of market close
        ],
        name="amount",
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected).all()
