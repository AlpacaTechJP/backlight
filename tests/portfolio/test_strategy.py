from backlight.portfolio.strategy import generate_simple_trades as module
import pytest
import pandas as pd
import numpy as np

import backlight
from backlight.asset.currency import Currency


@pytest.fixture
def strategy_name():
    return "simple_entry_and_exit"


@pytest.fixture
def strategy_params():
    return {"max_holding_time": pd.Timedelta("2min")}


@pytest.fixture
def signals():
    signals = []

    symbol = "usdjpy"
    currency_unit = Currency.JPY
    periods = 10
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
        ],
        columns=["up", "neutral", "down"],
    )
    signals.append(backlight.signal.from_dataframe(df, symbol, currency_unit))

    symbol = "eurjpy"
    currency_unit = Currency.JPY
    periods = 10
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
        ],
        columns=["up", "neutral", "down"],
    )
    signals.append(backlight.signal.from_dataframe(df, symbol, currency_unit))

    return signals


@pytest.fixture
def markets():
    markets = []
    symbol = "usdjpy"
    currency_unit = Currency.JPY
    periods = 10
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=np.arange(periods)[:, None],
        columns=["mid"],
    )
    markets.append(backlight.datasource.from_dataframe(df, symbol, currency_unit))

    symbol = "eurjpy"
    currency_unit = Currency.JPY
    periods = 10
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
        data=10 - np.arange(periods)[:, None],
        columns=["mid"],
    )
    markets.append(backlight.datasource.from_dataframe(df, symbol, currency_unit))
    return markets


def test_generate_simple_trades(markets, signals, strategy_name, strategy_params):
    generated_trades = module(markets, signals, strategy_name, strategy_params)

    expected = pd.DataFrame(
        index=markets[0].index,
        data=[1, -1, -1, 2, 1, -2, -2, 1, 1, 0],
        columns=["amount"],
    )

    for trades in generated_trades:
        assert (trades.amount == expected.amount).all()
