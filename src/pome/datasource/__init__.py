from ..query import query
from .marketdata import MarketData


def load_marketdata(symbol, start_dt, end_dt, url):
    df = query(symbol, start_dt, end_dt, url)
    return MarketData(df, symbol, start_dt, end_dt)
