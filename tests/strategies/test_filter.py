from backlight.strategies import filter as module
import pytest
import pandas as pd
import numpy as np

import backlight
import backlight.trades
from backlight.strategies.amount_based import simple_entry_and_exit
from backlight.asset.currency import Currency


@pytest.fixture
def symbol():
    return "USDJPY"


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


@pytest.fixture
def askbid(symbol, currency_unit):
    periods = 22
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=[[i + i % 3, i - i % 3] for i in range(periods)],
        columns=["ask", "bid"],
    )
    market = backlight.datasource.from_dataframe(df, symbol, currency_unit)
    return market


@pytest.fixture
def trades(market, signal):
    max_holding_time = pd.Timedelta("3min")
    trades = simple_entry_and_exit(market, signal, max_holding_time)
    return trades


def test_limit_max_amount(market, trades):
    max_amount = 2.0
    limited = module.limit_max_amount(trades, max_amount)
    expected = pd.DataFrame(
        index=market.index,
        data=[
            [True, 1.0],  # 1.0
            [True, -1.0],  # 0.0
            [False, 0.0],  # 0.0
            [True, 0.0],  # 0.0
            [True, 2.0],  # 2.0
            [True, -1.0],  # 1.0
            [True, -2.0],  # -1.0
            [True, -1.0],  # -2.0
            [True, 1.0],  # -1.0
            [True, 2.0],  # 1.0
            [True, 1.0],  # 2.0
            [False, 0.0],  # 2.0
            [True, -2.0],  # 0.0
            [True, -2.0],  # -2.0
            [False, 0.0],  # -2.0
            [True, 1.0],  # -1.0
            [True, 1.0],  # 0.0
            [False, 0.0],  # 0.0
            [True, 1.0],  # 1.0
            [True, 1.0],  # 2.0
            [False, 0.0],  # 2.0
            [True, -2.0],  # 0.0
        ],
        columns=["exist", "amount"],
    )
    assert (limited.amount == expected.amount[expected.exist]).all()


def test_skip_entry_by_spread(trades, askbid):
    spread = 2.0
    limited = module.skip_entry_by_spread(trades, askbid, spread)
    expected = pd.DataFrame(
        index=askbid.index,
        data=[
            [True, 1.0],  # 1.0
            [True, -1.0],  # 0.0
            [False, 0.0],  # 0.0
            [True, 0.0],  # 0.0
            [True, 2.0],  # 2.0
            [False, 0.0],  # 2.0
            [True, -2.0],  # 0.0
            [True, -1.0],  # -1.0
            [False, 0.0],  # -1.0
            [True, 2.0],  # 1.0
            [True, 1.0],  # 2.0
            [False, 0.0],  # 2.0
            [True, -2.0],  # 0.0
            [True, -2.0],  # -2.0
            [False, 0.0],  # 0.0
            [True, 1.0],  # -1.0
            [True, 1.0],  # -2.0
            [False, 0.0],  # 0.0
            [True, 1.0],  # 1.0
            [True, 1.0],  # 2.0
            [False, 0.0],  # 2.0
            [True, -2.0],  # 0.0
        ],
        columns=["exist", "amount"],
    )
    assert (limited.amount == expected.amount[expected.exist]).all()


def test_filter_entry_by_time(trades, symbol, currency_unit):
    result = module.filter_entry_by_time(trades, "minute", [1, 3, 8, 12])
    df = pd.DataFrame(
        data=[
            [1.0, 0.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
            [1.0, 2.0],
            [1.0, 1.0],
            [-1.0, 4.0],
            [-1.0, 2.0],
            [1.0, 4.0],
            [1.0, 6.0],
            [-1.0, 6.0],
            [-1.0, 9.0],
            [1.0, 9.0],
        ],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2018-06-06 00:00:00"),
                pd.Timestamp("2018-06-06 00:01:00"),
                pd.Timestamp("2018-06-06 00:03:00"),
                pd.Timestamp("2018-06-06 00:03:00"),
                pd.Timestamp("2018-06-06 00:04:00"),
                pd.Timestamp("2018-06-06 00:05:00"),
                pd.Timestamp("2018-06-06 00:06:00"),
                pd.Timestamp("2018-06-06 00:08:00"),
                pd.Timestamp("2018-06-06 00:09:00"),
                pd.Timestamp("2018-06-06 00:12:00"),
                pd.Timestamp("2018-06-06 00:12:00"),
                pd.Timestamp("2018-06-06 00:15:00"),
            ]
        ),
        columns=["amount", "_id"],
    )

    expected = backlight.trades.trades.from_dataframe(df, symbol, currency_unit)
    assert (result.all() == expected.all()).all()


@pytest.fixture
def hourly_trades(symbol, currency_unit):
    data = [
        1.0,  # entry at UTC 0
        -2.0,
        1.0,  # entry at UTC 2
        2.0,
        -4.0,  # entry at UTC 4
        2.0,
        1.0,  # entry at UTC 6
        0.0,
        1.0,  # entry at UTC 8
        0.0,
    ]
    index = pd.date_range(start="2018-06-06", freq="1H", periods=len(data))
    trades = []
    for i in range(0, len(data), 2):
        trade = pd.Series(index=index[i : i + 2], data=data[i : i + 2], name="amount")
        trades.append(trade)
    trades = backlight.trades.make_trades(symbol, trades, currency_unit)
    return trades


def test_skip_entry_by_hours(hourly_trades):
    hours = [2, 5, 6, 7]
    limited = module.skip_entry_by_hours(hourly_trades, hours)
    expected = pd.concat(
        [
            hourly_trades.get_trade(0),
            hourly_trades.get_trade(2),
            hourly_trades.get_trade(4),
        ],
        axis=0,
    )
    assert (limited.amount == expected).all()
