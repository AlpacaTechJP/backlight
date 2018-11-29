import pandas as pd
from typing import List, Tuple

from backlight import datasource
from backlight.datasource.marketdata import MarketData
from backlight.trades.trades import Trade, Trades, Transaction  # noqa


def _sum(a: list) -> int:
    return sum(a) if len(a) != 0 else 0


def make_trades(trades: List[Trade], mkt: MarketData) -> Trades:
    t = Trades(mkt)
    t._trades = trades
    t.reset()

    assert t.symbol == mkt.symbol

    return t


def _pl(trade: Trade, mkt: MarketData) -> float:
    amount = trade.amount.cumsum()
    mkt, amount = mkt.align(amount, axis=0, join="inner")
    # TODO: ask bid pricing
    next_price = mkt["mid"].shift(periods=-1)
    price_diff = next_price - mkt["mid"]
    pl = (price_diff * amount).shift(periods=1)[1:]  # drop first nan
    return _sum(pl)


def count(trades: Trades) -> Tuple[int, int, int]:
    mkt = datasource.from_dataframe(trades, trades.symbol)
    pls = [_pl(t, mkt) for t in trades.trades]
    total = len(pls)
    win = _sum([pl > 0.0 for pl in pls])
    lose = _sum([pl < 0.0 for pl in pls])
    return total, win, lose
