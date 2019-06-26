import pytest

import pandas as pd
import numpy as np

from backlight.positions.positions import Positions
from backlight.portfolio.portfolio import Portfolio


@pytest.fixture
def positions():
    periods = 4
    ps = []

    for symbol in ["usdjpy", "eurusd", "usdjpy", "usdjpy", "eurjpy", "eurusd"]:
        df = pd.DataFrame(
            index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
            data=np.arange(3 * periods).reshape(periods, 3),
            columns=["amount", "price", "principal"],
        )
        p = Positions(df)
        p.symbol = symbol

        ps.append(p)

    return ps


def test_init_portfolio(positions):
    periods = 4
    expected = []
    k = 3

    for symbol in ["usdjpy", "eurusd", "eurjpy"]:
        df = pd.DataFrame(
            index=pd.date_range(start="2018-06-06", freq="1min", periods=periods),
            data=np.arange(0, 3 * periods * k, k).reshape(periods, 3),
            columns=["amount", "price", "principal"],
        )
        p = Positions(df)
        p.symbol = symbol

        expected.append(p)

        k -= 1

    pf = Portfolio(positions)

    assert len(pf._positions) == len(expected)

    for (position, exp) in zip(pf._positions, expected):

        assert ((exp == position).all()).all()
