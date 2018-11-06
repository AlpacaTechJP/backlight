from backlight.query import query
from backlight.datasource.marketdata import MarketData


def load_marketdata(symbol, start_dt, end_dt, url):
    """An abstraction interface for loading the market data.

    Args:
        symbol (str): symbol to query
        start_dt (pd.Timestamp):  query from
        end_dt (pd.Timestamp):  query to
        url (str):  an url to the data source

    Returns:
        MarketData
    """
    df = query(symbol, start_dt, end_dt, url)
    return from_dataframe(symbol, df)


def from_dataframe(df, symbol, col_mapping=None):
    """Create a MarketData instance out of a DataFrame object

    Args:
        df (pd.DataFrame):  DataFrame
        symbol (str): symbol to query
        col_mapping (dict): A dict to map columns

    Returns:
        MarketData
    """
    mkt = MarketData(df)
    mkt.symbol = symbol
    return mkt
