from typing import List
from backlight.datasource.marketdata import MarketData
from backlight.trades.trades import Transaction, Trade, Trades  # noqa


def make_trades(trades: List[Trade], mkt: MarketData) -> Trades:
    """
    Order of concat is important, if mkt in front of trades,
    it will have error since it will use the type of mkt(MarketData)
    to concat, but trades is normal DataFrame.
    """
    t = Trades(mkt)
    t.trades = trades
    return t
