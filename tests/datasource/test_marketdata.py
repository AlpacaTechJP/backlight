from backlight.datasource import marketdata as module
from unittest import mock
import pandas as pd
import os


@mock.patch.dict(os.environ, {"TICK_MARKETSTORE_HOST": "8888"})
def test_MarketData():
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=3),
        data=[[0, 2], [2, 4], [4, 6]],
        columns=["ask", "bid"],
    )
    md = module.MarketData(df, "ABC", df.index[0], df.index[-1])
    assert md.symbol == "ABC"
    assert all(md.mid.values == [1, 3, 5])

    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=3),
        data=[0, 2, 6],
        columns=["mid"],
    )
    md = module.MarketData(df, "ABC", df.index[0], df.index[-1])
    assert all(md.mid.values == [0, 2, 6])
