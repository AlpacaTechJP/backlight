import pandas as pd
from typing import Optional

from backlight.datasource.marketdata import MarketData, MidMarketData, AskBidMarketData
from backlight.query import query


def load_marketdata(
    symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, url: str
) -> MarketData:
    """An abstraction interface for loading the market data.

    Args:
        symbol :  A symbol to query
        start_dt :  Date to query from
        end_dt :  Date to query to
        url :  An url to the data source

    Returns:
        MarketData
    """
    df = query(symbol, start_dt, end_dt, url)
    return from_dataframe(df, symbol, col_mapping=None)


def from_dataframe(
    df: pd.DataFrame, symbol: str, col_mapping: Optional[dict] = None
) -> MarketData:
    """Create a MarketData instance out of a DataFrame object

    Args:
        df :  DataFrame
        symbol :  A symbol to query
        col_mapping :  A dict to map columns

    Returns:
        MarketData
    """
    df = df.copy()

    if col_mapping is not None:
        df = df.rename(columns=col_mapping)

    mkt = None

    if ("ask" in df.columns) and ("bid" in df.columns):
        from backlight.datasource.marketdata import AskBidMarketData

        mkt = AskBidMarketData(df)
    elif "mid" in df.columns:
        from backlight.datasource.marketdata import MidMarketData

        mkt = MidMarketData(df)

    if mkt is None:
        raise ValueError("Unsupported marketdata")

    mkt.symbol = symbol
    mkt.reset_cols()

    return mkt


def mid2askbid(mkt: MidMarketData, spread: float) -> AskBidMarketData:
    """Convert MidMarketData to AskBidMarketData.

    Args:
        mkt: MidMarketData
        spread: ask/bid spread from mid price.
    """
    mkt.loc[:, "ask"] = mkt.mid + spread
    mkt.loc[:, "bid"] = mkt.mid - spread
    return from_dataframe(mkt, mkt.symbol)
