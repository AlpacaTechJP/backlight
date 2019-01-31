from backlight.strategies import exit as module
import pytest
import pandas as pd
import numpy as np

import backlight
from backlight.labelizer.common import TernaryDirection
from backlight.strategies.common import Action
from backlight.strategies.entry import direction_based_entry
from backlight.trades.trades import Transaction, make_trades, make_trade


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


@pytest.fixture
def entries(signal, market):
    direction_action_dict = {
        TernaryDirection.UP: Action.TakeLong,
        TernaryDirection.NEUTRAL: Action.Donothing,
        TernaryDirection.DOWN: Action.TakeShort,
    }
    entries = direction_based_entry(market, signal, direction_action_dict)
    return entries


def test_exit_at_max_holding_time(market, signal, entries):
    max_holding_time = pd.Timedelta("3min")
    trades = module.exit_at_max_holding_time(market, signal, entries, max_holding_time)
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
    assert (trades.amount == expected.amount[expected.exist]).all()


def test_exit_by_trailing_stop(market, signal, entries):
    symbol = "usdjpy"
    data = [
        [1.0],  # 00:00:00
        [2.0],  # 00:01:00
        [3.0],  # 00:02:00
        [4.0],  # 00:03:00
        [5.0],  # 00:04:00
        [4.0],  # 00:05:00
        [3.0],  # 00:06:00
        [2.0],  # 00:07:00
        [1.0],  # 00:08:00
        [0.0],  # 00:09:00
        [1.0],  # 00:10:00
    ]
    periods = len(data)
    market = backlight.datasource.from_dataframe(
        pd.DataFrame(
            index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
            data=data,
            columns=["mid"],
        ),
        symbol,
    )
    entries = make_trades(
        symbol,
        (
            make_trade([Transaction(pd.Timestamp("2018-06-06 00:00:00"), 1.0)]),
            make_trade([Transaction(pd.Timestamp("2018-06-06 00:00:00"), -1.0)]),
            make_trade([Transaction(pd.Timestamp("2018-06-06 00:00:00"), 0.0)]),
            make_trade([Transaction(pd.Timestamp("2018-06-06 00:03:00"), 1.0)]),
            make_trade([Transaction(pd.Timestamp("2018-06-06 00:03:00"), 0.5)]),
            make_trade([Transaction(pd.Timestamp("2018-06-06 00:03:00"), -1.0)]),
        ),
    )

    initial_stop = 2.0
    trailing_stop = 1.0
    trades = module.exit_by_trailing_stop(market, entries, initial_stop, trailing_stop)
    expected = make_trades(
        symbol,
        (
            make_trade(
                [
                    Transaction(pd.Timestamp("2018-06-06 00:00:00"), 1.0),
                    Transaction(
                        pd.Timestamp("2018-06-06 00:05:00"), -1.0
                    ),  # trail stop
                ]
            ),
            make_trade(
                [
                    Transaction(pd.Timestamp("2018-06-06 00:00:00"), -1.0),
                    Transaction(pd.Timestamp("2018-06-06 00:02:00"), 1.0),  # loss cut
                ]
            ),
            make_trade([Transaction(pd.Timestamp("2018-06-06 00:00:00"), 0.0)]),
            make_trade(
                [
                    Transaction(pd.Timestamp("2018-06-06 00:03:00"), 1.0),
                    Transaction(
                        pd.Timestamp("2018-06-06 00:05:00"), -1.0
                    ),  # trail stop
                ]
            ),
            make_trade(
                [
                    Transaction(pd.Timestamp("2018-06-06 00:03:00"), 0.5),
                    Transaction(pd.Timestamp("2018-06-06 00:05:00"), -0.5),  # loss cut
                ]
            ),
            make_trade(
                [
                    Transaction(pd.Timestamp("2018-06-06 00:03:00"), -1.0),
                    Transaction(pd.Timestamp("2018-06-06 00:10:00"), 1.0),  # trail stop
                ]
            ),
        ),
    )

    pd.testing.assert_frame_equal(trades, expected)
