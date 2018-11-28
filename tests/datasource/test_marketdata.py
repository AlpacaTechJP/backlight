from backlight.datasource import marketdata as module
import pandas as pd


def test_MarketData():
    start_dt = pd.Timestamp("2018-06-06")
    end_dt = pd.Timestamp("2018-06-08")
    df = pd.DataFrame(
        index=pd.date_range(start=start_dt, periods=3),
        data=[[0, 2], [2, 4], [4, 6]],
        columns=["ask", "bid"],
    )
    md = module.MarketData(df)
    assert md.start_dt == start_dt
    assert md.end_dt == end_dt


def test_AskBidMarketData():
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=3),
        data=[[0, 2], [2, 4], [4, 6]],
        columns=["ask", "bid"],
    )
    md = module.AskBidMarketData(df)
    assert all(md.mid.values == [1, 3, 5])


def test_MidMarketData():
    df = pd.DataFrame(
        index=pd.date_range(start="2018-06-06", periods=3),
        data=[0, 2, 6],
        columns=["mid"],
    )
    md = module.MidMarketData(df)
    assert all(md.mid.values == [0, 2, 6])
