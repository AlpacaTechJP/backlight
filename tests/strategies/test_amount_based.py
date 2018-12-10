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


def test_direction_based_trades(market, signal):
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    trades = module.direction_based_trades(market, signal, direction_action_dict)
    expected = pd.Series(
        index=market.index,
        data=[
            1.0,  # U
            -1.0,  # D
            0.0,  # N
            1.0,  # U
            1.0,  # U
            -1.0,  # D
            -1.0,  # D
            0.0,  # N
            0.0,  # N
            1.0,  # U
            1.0,  # U
            1.0,  # U
            -1.0,  # D
            -1.0,  # D
            -1.0,  # D
            0.0,  # N
            0.0,  # N
            0.0,  # N
            1.0,  # U
            1.0,  # U
            1.0,  # U
            1.0,  # U
        ],
        name="amount",
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected).all()


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
    expected = pd.DataFrame(
        index=market.index,
        data=[
            [True, 1.0],  # U
            [True, -1.0],  # D
            [False, 0.0],  # N
            [True, 0.0],  # U + D
            [True, 2.0],  # U + U
            [True, -1.0],  # D + N
            [True, -2.0],  # D + D
            [True, -1.0],  # N + D
            [True, 1.0],  # N + U
            [True, 2.0],  # U + U
            [True, 1.0],  # U + N
            [True, 1.0],  # U + N
            [True, -2.0],  # D + D
            [True, -2.0],  # D + D
            [True, -2.0],  # D + D
            [True, 1.0],  # N + U
            [True, 1.0],  # N + U
            [True, 1.0],  # N + U
            [True, 1.0],  # U + N
            [True, 1.0],  # U + N
            [True, 1.0],  # U + N
            [True, -3.0],  # U + 4D because of market close
        ],
        columns=["exist", "amount"],
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected.amount[expected.exist]).all()


def test_simple_entry(market, signal):
    max_holding_time = pd.Timedelta("3min")
    trades = module.simple_entry(market, signal, max_holding_time)
    expected = pd.DataFrame(
        index=market.index,
        data=[
            [True, 1.0],  # U
            [True, -1.0],  # D
            [False, 0.0],  # N
            [True, 0.0],  # U + D
            [True, 2.0],  # U + U
            [True, -1.0],  # D + N
            [True, -2.0],  # D + D
            [True, -1.0],  # N + D
            [True, 1.0],  # N + U
            [True, 2.0],  # U + U
            [True, 1.0],  # U + N
            [True, 1.0],  # U + N
            [True, -2.0],  # D + D
            [True, -2.0],  # D + D
            [True, -2.0],  # D + D
            [True, 1.0],  # N + U
            [True, 1.0],  # N + U
            [True, 1.0],  # N + U
            [True, 1.0],  # U + N
            [True, 1.0],  # U + N
            [True, 1.0],  # U + N
            [True, -3.0],  # U + 4D because of market close
        ],
        columns=["exist", "amount"],
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected.amount[expected.exist]).all()


def test_only_entry_short(market, signal):
    max_holding_time = pd.Timedelta("3min")
    trades = module.only_entry_short(market, signal, max_holding_time)
    expected = pd.DataFrame(
        index=market.index,
        data=[
            [False, 0.0],  # N
            [True, -1.0],  # D
            [False, 0.0],  # N
            [False, 0.0],  # N + N
            [True, 1.0],  # N + U
            [True, -1.0],  # D + N
            [True, -1.0],  # D + N
            [False, 0.0],  # N + N
            [True, 1.0],  # N + U
            [True, 1.0],  # N + U
            [False, 0.0],  # N + N
            [False, 0.0],  # N + N
            [True, -1.0],  # D + N
            [True, -1.0],  # D + N
            [True, -1.0],  # D + N
            [True, 1.0],  # N + U
            [True, 1.0],  # N + U
            [True, 1.0],  # N + U
            [False, 0.0],  # N + N
            [False, 0.0],  # N + N
            [False, 0.0],  # N + N
            [False, 0.0],  # N + 4N because of market close
        ],
        columns=["exist", "amount"],
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected.amount[expected.exist]).all()


def test_only_entry_long(market, signal):
    max_holding_time = pd.Timedelta("3min")
    trades = module.only_entry_long(market, signal, max_holding_time)
    expected = pd.DataFrame(
        index=market.index,
        data=[
            [True, 1.0],  # U
            [False, 0.0],  # N
            [False, 0.0],  # N
            [True, 0.0],  # U + D
            [True, 1.0],  # U + N
            [False, 0.0],  # N + N
            [True, -1.0],  # N + D
            [True, -1.0],  # N + D
            [False, 0.0],  # N + N
            [True, 1.0],  # U + N
            [True, 1.0],  # U + N
            [True, 1.0],  # U + N
            [True, -1.0],  # N + D
            [True, -1.0],  # N + D
            [True, -1.0],  # N + D
            [False, 0.0],  # N + N
            [False, 0.0],  # N + N
            [False, 0.0],  # N + N
            [True, 1.0],  # U + N
            [True, 1.0],  # U + N
            [True, 1.0],  # U + N
            [True, -3.0],  # U + 4D because of market close
        ],
        columns=["exist", "amount"],
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected.amount[expected.exist]).all()


def test_exit_opposite_signal(market, signal):
    max_holding_time = pd.Timedelta("3min")
    trades = module.exit_on_oppsite_signals(market, signal, max_holding_time)
    expected = pd.DataFrame(
        index=market.index,
        data=[
            [True, 1.0],  # U
            [True, -2.0],  # D + D
            [False, 0.0],  # N
            [True, 2.0],  # U + U
            [True, 1.0],  # U
            [True, -3.0],  # D + 2D
            [True, -1.0],  # D
            [False, 0.0],  # N
            [True, 1.0],  # N + U
            [True, 2.0],  # U + U
            [True, 1.0],  # U
            [True, 1.0],  # U
            [True, -4.0],  # D + 3D
            [True, -1.0],  # D
            [True, -1.0],  # D
            [True, 1.0],  # N + U
            [True, 1.0],  # N + U
            [True, 1.0],  # N + U
            [True, 1.0],  # U
            [True, 1.0],  # U
            [True, 1.0],  # U
            [True, -3.0],  # U + 4D because of market close
        ],
        columns=["exist", "amount"],
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected.amount[expected.exist]).all()


def test_exit_other_signal(market, signal):
    max_holding_time = pd.Timedelta("3min")
    trades = module.exit_on_other_signals(market, signal, max_holding_time)
    expected = pd.DataFrame(
        index=market.index,
        data=[
            [True, 1.0],  # U
            [True, -2.0],  # D + D
            [True, 1.0],  # N + U
            [True, 1.0],  # U
            [True, 1.0],  # U
            [True, -3.0],  # D + 2D
            [True, -1.0],  # D
            [True, 2.0],  # N + 2U
            [False, 0.0],  # N
            [True, 1.0],  # U
            [True, 1.0],  # U
            [True, 1.0],  # U
            [True, -4.0],  # D + 3D
            [True, -1.0],  # D
            [True, -1.0],  # D
            [True, 3.0],  # N + 3U
            [False, 0.0],  # N
            [False, 0.0],  # N
            [True, 1.0],  # U
            [True, 1.0],  # U
            [True, 1.0],  # U
            [True, -3.0],  # U + 4D because of market close
        ],
        columns=["exist", "amount"],
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected.amount[expected.exist]).all()


def test_exit_by_expectation(market):
    symbol = "usdjpy"
    data = [
        [1, 0, 0],  # expectation = 1
        [0, 0, 1],  # expectation = -1
        [0, 0, 0],  # expectation = 0
        [1, 0.9, 0.9],  # expectation = 0.1
        [1, 0.0, 0.9],  # expectation = 0.1
        [0.9, 0.9, 1],  # expectation = -0.1
        [0.9, 0.0, 1],  # expectation = -0.1
        [0, 0.9, 0],  # expectation = 0.0
    ]
    periods = len(data)
    signal = backlight.signal.from_dataframe(
        pd.DataFrame(
            index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
            data=data,
            columns=["up", "neutral", "down"],
        ),
        symbol,
    )

    max_holding_time = pd.Timedelta("3min")
    trades = module.exit_by_expectation(market, signal, max_holding_time)
    expected = pd.DataFrame(
        index=signal.index,
        data=[
            [True, 1.0],  # U
            [True, -2.0],  # D + D
            [False, 0.0],  # N
            [True, 2.0],  # U + U
            [True, 1.0],  # U
            [True, -3.0],  # D + 2D
            [True, -1.0],  # D
            [True, 2.0],  # N + 2U
        ],
        columns=["exist", "amount"],
    )
    trade = backlight.trades.flatten(trades)
    assert (trade.amount == expected.amount[expected.exist]).all()
