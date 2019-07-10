from backlight.plot import plot_positions as module

import pandas as pd
import matplotlib
import pytest

import backlight.datasource
import backlight.positions
from backlight.trades.trades import make_trades
from backlight.asset.currency import Currency


@pytest.fixture
def positions():
    symbol = "usdjpy"
    currency_unit = Currency.JPY

    data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [9.0]]
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", freq="1min", periods=len(data)),
        data=data,
        columns=["mid"],
    )
    market = backlight.datasource.from_dataframe(df, symbol, currency_unit)

    data = [1.0, -2.0, 1.0, 2.0, -4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
    index = pd.date_range(start="2018-06-06", freq="1min", periods=len(data))
    trades = []
    for i in range(0, len(data), 2):
        trade = pd.Series(index=index[i : i + 2], data=data[i : i + 2], name="amount")
        trades.append(trade)
    trades = make_trades(symbol, trades, currency_unit)

    return backlight.positions.calc_positions(trades, market)


def test_plot_pl(positions):
    matplotlib.use("agg")
    module.plot_pl(positions)


def test_plot_cumulative_pl(positions):
    matplotlib.use("agg")
    module.plot_cumulative_pl(positions)
