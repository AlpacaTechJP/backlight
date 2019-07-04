import pandas as pd
import numpy as np
from typing import List
from functools import reduce

from backlight.positions.positions import Positions
import backlight.positions.positions
from backlight.datasource.marketdata import MarketData
from backlight.metrics.position_metrics import calc_pl
from backlight.positions import calc_positions
from backlight.trades.trades import Trades, from_dataframe
from joblib import Parallel, delayed


class Portfolio:
    """
    An abstract definition of a Portfolio is defined as a list of Positions
    Each element of the list represent the positions of an asset that contribute to the portfolio
    """

    def __init__(self, positions: List[Positions]):
        """
        If there is positions with the same symbol, their value are sum
        """
        symbols = [position.symbol for position in positions]
        assert len(symbols) == len(set(symbols))
        self._positions = positions

    def value(self) -> pd.DataFrame:
        """ DataFrame of the portfolio valuation of each asset"""
        pl = pd.DataFrame()
        for p in self._positions:
            # Compute PL of positions of each asset
            pl[p.symbol] = calc_pl(p)
        return pl

    def get_amount(self, symbol: str) -> pd.Series:
        """ Return amounts of each asset in the portfolio at each time step"""
        for p in self._positions:
            if p.symbol == symbol:
                return p.amount
        raise ValueError("Passed symbol not found in portfolio")


def calculate_lots_size(
    mkt: List[MarketData], principal: List[float], max_amount: List[int]
) -> List[int]:
    """
    Compute lot_size based on the principal, max_amount and the makrtdata.
    lot = (Principal/max_amount) / (market_price_at_0)

    """
    lots = []
    for (m, p, amount) in zip(mkt, principal, max_amount):
        lots.append(int((p / amount) / m.mid[0]))
    return lots


def construct_portfolio(
    trades: List[Trades],
    mkt: List[MarketData],
    principal: List[float],
    lot_size: List[int],
) -> Portfolio:
    """
    Take a list of Trades and MarketData and return a portfolio
    args:
        - trades : list of unit trades (1 or -1 for th amount)
        - mkt : list of mkt data
        - principal: list of principal per asset
        - lot_size : list of lot sizes per asset
                    (e.g. trade.amount = 1 is equivalent to buying 1*lot_size assets)
    return:
        Portfolio
    """

    assert len(principal) == len(trades)
    assert len(lot_size) == len(trades)

    symbols2mkt = {m.symbol: m for m in mkt}
    symbols = [t.symbol for t in trades]
    assert set(symbols) == set(symbols2mkt.keys())

    # Transform trades following the lot_size
    mult_trades = []
    for (trade, lot) in zip(trades, lot_size):
        mult_trade = trade.copy()
        mult_trade["amount"] *= lot
        mult_trades.append(mult_trade)

    # Construct positions and return Portfolio
    positions = Parallel(n_jobs=-1, max_nbytes=None)(
        [
            delayed(calc_positions)(
                trade, symbols2mkt[trade.symbol], principal=principal_per_asset
            )
            for (trade, principal_per_asset) in zip(mult_trades, principal)
        ]
    )

    symbols = [p.symbol for p in positions]
    if len(set(symbols)) != len(symbols):
        positions = _fusion_positions(positions)

    return Portfolio(positions)


def _fusion_positions(positions: List[Positions]) -> List[Positions]:
    """
    Take a list of Positions and sum those with the same symbols
    args :
        - positions : a list of Positions to fusion
    """

    unique_positions = []
    symbols = [p.symbol for p in positions]
    columns = positions[0].columns

    for symbol in sorted(set(symbols)):
        positions_of_symbol = [p for p in positions if p.symbol == symbol]

        indices = [p.index for p in positions if p.symbol == symbol]
        union_index = indices[0].union_many(indices)

        dfs = [p for p in positions if p.symbol == symbol]
        df = reduce(lambda x, y: x.add(y, fill_value=0), dfs)

        position = backlight.positions.positions.from_dataframe(df, symbol)
        unique_positions.append(position)

    return unique_positions


def calculate_pl(pt: Portfolio, mkt: List[MarketData]) -> pd.DataFrame:
    """
    Apply the sum on the homogenized portfolio
    args:
        - portfolio : a defined portfolio
        - mkt : list of marketdata for each asset
        - base_ccy : asset of reference
    
    """
    pl = pt.value()
    return pl.sum(axis=1)
