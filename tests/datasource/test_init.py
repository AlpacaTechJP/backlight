from backlight import datasource as module
import pandas as pd


def test_from_dataframe():
    symbol = "USDJPY"
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=3),
        data=[[0, 2], [2, 4], [4, 6]],
        columns=["ask", "bid"],
    )
    mkt = module.from_dataframe(df, symbol)
    assert mkt.symbol == symbol
    assert all(mkt.mid.values == [1, 3, 5])

    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=3),
        data=[[0, 2], [2, 4], [4, 6]],
        columns=["aaa", "bbb"],
    )
    col_mapping = {"aaa": "ask", "bbb": "bid"}
    mkt = module.from_dataframe(df, symbol, col_mapping=col_mapping)
    assert mkt.symbol == symbol
    assert all(mkt.mid.values == [1, 3, 5])
