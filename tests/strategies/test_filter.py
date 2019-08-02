from backlight.strategies import filter as module
import pytest
import pandas as pd
import numpy as np

import backlight
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
    """
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
            [True, 1.0],  # 3.0
            [True, -2.0],  # 1.0
            [True, -2.0],  # -1.0
            [True, -2.0],  # -3.0
            [True, 1.0],  # -2.0
            [True, 1.0],  # -1.0
            [True, 1.0],  # 0.0
            [True, 1.0],  # 1.0
            [True, 1.0],  # 2.0
            [True, 1.0],  # 3.0
            [True, -3.0],  # 0.0
        ],
        columns=["exist", "amount"],
    )
    """
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
    print(limited.amount)
    assert (limited.amount == expected.amount[expected.exist]).all()
