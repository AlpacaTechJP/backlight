import pandas as pd

from backlight.datasource.marketdata import MarketData
from backlight.query import query


def load_marketdata(
    symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, url: str
) -> MarketData:
    """An abstraction interface for loading the market data.

    Args:
        symbol :  symbol to query
        start_dt :  query from
        end_dt :  query to
        url :  an url to the data source

    Returns:
        MarketData
    """
    df = query(symbol, start_dt, end_dt, url)
    return from_dataframe(symbol, df)


def from_dataframe(
    df: pd.DataFrame, symbol: str, col_mapping: dict = None
) -> MarketData:
    """Create a MarketData instance out of a DataFrame object

    Args:
        df :  DataFrame
        symbol :  symbol to query
        col_mapping :  A dict to map columns

    Returns:
        MarketData
    """
    mkt = MarketData(df)
    mkt.symbol = symbol
    return mkt
